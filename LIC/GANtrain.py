import os
import random
import shutil
import sys

import torch
import torch.nn as nn
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from utils.meter import AverageMeter
from models.generator import Generator
from models.discriminator import Discriminator
from models.compressor import Compressor


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="dataset/lic",
        help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--d-learning-rate",
        default=2.5e-4,
        type=float,
        help="discriminator learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--g-learning-rate",
        default=2.5e-4,
        help="generator learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def configure_optimizer(G, D, args):
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_learning_rate)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.g_learning_rate)

    return g_optimizer, d_optimizer


def train_one_epoch(epoch, G, D, C, g_optim, d_optim, train_dataloader, device, criterion, mse, train_with_D):

    drop = 0.25
    bias = 0.35

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        output = C.compress(d)
        corrupted_data, gt_data, masks, inverted_masks = C.drop(drop, output["strings"], output["shape"])
        gauss = torch.normal(0, 0.2, corrupted_data.shape).to(device)
        fake_gt_data = G(corrupted_data * inverted_masks + gauss * masks)

        fake_in = corrupted_data * inverted_masks + fake_gt_data * masks
        real_in = corrupted_data * inverted_masks + gt_data * masks

        # update d
        if train_with_D:
            fake_gt_data_D = D(fake_in)
            real_gt_data_D = D(real_in)

            targets_real = torch.ones(real_gt_data_D.shape).to(device)
            targets_fake = torch.zeros(fake_gt_data_D.shape).to(device)

            d_optim.zero_grad()

            # fake_loss = criterion(fake_gt_data_D, targets_fake)
            # real_loss = criterion(real_gt_data_D, targets_real)

            fake_loss = criterion(fake_gt_data_D, targets_fake + torch.normal(0.1, 0.05, targets_fake.shape).to(device))
            real_loss = criterion(real_gt_data_D,
                                  targets_real + torch.normal(-0.1, 0.05, targets_real.shape).to(device))

            d_loss = (fake_loss + real_loss) / 2
            d_loss.backward(retain_graph=True)

            d_optim.step()

        # update g
        mse_loss = mse(gt_data * masks, fake_gt_data * masks)
        # g_loss = lmbda * 255 ** 2 * mse_loss
        g_loss = mse_loss * (1 - bias)
        if train_with_D:
            outputs = D(fake_in)
            bias_loss = criterion(outputs, targets_real)
            g_loss += bias_loss * bias

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if i % 20 == 0:
            if train_with_D:
                print(
                    "Epoch %d. Iteration %d. Generator loss: %f (mse: %f, bias: %f). Discriminator loss: %f (real: %f. fake: %f)" % (
                        epoch + 1, i, g_loss.item(), mse_loss.item(), bias_loss.item(), d_loss.item(),
                        real_loss.item(),
                        fake_loss.item()))
            else:
                print("Epoch %d. Iteration %d. Generator loss: %f (mse: %f, bias: %f)." % (
                    epoch + 1, i, g_loss.item(), mse_loss.item(), bias_loss.item()))


def demo_epoch(epoch, G, C, test_dataloader, device, mse):
    loss = AverageMeter()
    for i, d in enumerate(test_dataloader):
        output = C.compress(d)
        corrupted_data, gt_data, masks, inverted_masks = C.drop(0.25, output["strings"], output["shape"])
        with torch.no_grad:
            fake_gt_data = G(corrupted_data)

            mse_loss = mse(gt_data * masks, fake_gt_data * masks)
            g_loss = mse_loss
            loss.update(g_loss)

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
    )

    return loss.avg


def main(argv):
    device = "cuda"
    args = parse_args(argv)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    G = Generator()
    D = Discriminator()
    C = Compressor()

    D = D.to(device)
    G = G.to(device)
    C = C.to(device)

    checkpoint = torch.load('./checkpoints/models_LIC_1e-3.pth.tar', map_location=device)
    C.load_state_dict(checkpoint["state_dict"])
    C.eval()
    C.update()

    criterion = nn.BCELoss()
    mse = nn.MSELoss(reduction='mean')
    mae = nn.L1Loss(reduction='mean')

    criterion = criterion.to(device)
    mse = mse.to(device)

    save_path = "./saves/models_DG_1e-2.pth"
    best_save_path = "./saves/models_DG_best.pth"
    restore = True
    restore_path = "./saves/models_DG_1e-2.pth"

    g_optimizer, d_optimizer = configure_optimizer(G, D, args)

    last_epoch = 0
    if args.checkpoint:
        print("loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch" + 1]
        G.load_state_dict(checkpoint["g_state_dict"])
        D.load_state_dict(checkpoint["d_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer"])

    best_loss = float("inf")
    train_with_D = True
    for epoch in range(last_epoch, args.epochs):
        # train_with_D = not train_with_D
        # if epoch % 2 == 0 and epoch != 0:
        #     train_with_D = True
        train_one_epoch(epoch, G, D, C,
                        g_optimizer,
                        d_optimizer,
                        train_dataloader,
                        device,
                        criterion,
                        mse,
                        train_with_D=train_with_D)

        # loss = demo_epoch(epoch, G, args.test_dataset, device, mse)
        # is_best = loss < best_loss
        # best_loss = min(loss, best_loss)

        if args.save:
            torch.save({
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'Dopt_state_dict': d_optimizer.state_dict(),
                'Gopt_state_dict': g_optimizer.state_dict(),
            }, "models_DG_1e-3.pth")
            # if is_best:
            #     shutil.copyfile(save_path, best_save_path)
            print("Models saved.")


if __name__ == "__main__":
    main(sys.argv[1:])
