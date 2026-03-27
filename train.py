import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"   # prevents Qt display issues on headless systems

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before importing pyplot

import time
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Config import cfg
# from library.Dataloader import get_training_set
# from library.minimization_dataloader.temporal_subsampling import UCSD_Ped_dataset
from library.minimization_dataloader.temporal_subsampling_ROI import UCSD_Ped_dataset
from models.model import ConvLSTMAE


######------------- Training Epoch -------------######
def train_epoch(model, train_dataloader, criterion, optimizer, use_cuda, writer, epoch):
    print(f"Train Epoch {epoch}")

    losses = []
    model.train()

    for i, batch in enumerate(train_dataloader):
        if isinstance(batch, (list, tuple)):
            clips = batch[0]
        else:
            clips = batch

        if use_cuda:
            clips = clips.cuda(non_blocking=True)
            # clips = clips.permute(0,1,4,2,3)  # ensure channels last for UCSD data

        optimizer.zero_grad()

        recon_clip = model(clips)
        loss = criterion(recon_clip, clips)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0:
            print(f"Training Epoch {epoch}, Batch {i}, Loss: {np.mean(losses):.5f}", flush=True)

    epoch_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    print(f"Training Epoch: {epoch}, Loss: {epoch_loss:.4f}", flush=True)

    writer.add_scalar("Training Loss", epoch_loss, epoch)

    del clips, recon_clip, loss
    if use_cuda:
        torch.cuda.empty_cache()

    return epoch_loss


#####-----------------plot loss-------------------------#####

def plot_loss(train_loss, plot_file):
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def save_checkpoint(model, optimizer, epoch, train_loss_epoch, save_file_path):
    states = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss_epoch,
    }
    torch.save(states, save_file_path)

######------------- Training Epoch -------------######

def train(devices):
    use_cuda = torch.cuda.is_available()

    os.makedirs(os.path.join(cfg.logs, cfg.run_id), exist_ok=True)
    os.makedirs(os.path.join(cfg.saved_models_dir, cfg.run_id), exist_ok=True)

    writer = SummaryWriter(os.path.join(cfg.logs, cfg.run_id))

    model = ConvLSTMAE(in_channels=1)      
    
    # train_dataset = get_training_set(mode = cfg.mode)  # for UCSD Ped1
    train_dataset = UCSD_Ped_dataset(root_dir=cfg.UCSD_train_path,
                                     data_split="Train",
                                     shuffle=True,
                                     sequence_size=10,
                                     stride=cfg.stride,
                                     step=cfg.step,
                                     mode=cfg.mode,
                                     privacy=cfg.privacy,
                                     bg_history=100,
                                     bg_var_threshold=16,
                                     bg_warmup=20)

    print("Training clips:", len(train_dataset))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.batch,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    criterion = nn.MSELoss()
    
    device_name = f'cuda:{devices[0]}'
    print(f'Device name is {device_name}')
    if len(devices) > 1:
        print(f"Using multiple GPUs: {devices}")
        model = nn.DataParallel(model, device_ids=devices).cuda()
        criterion = criterion.cuda()
    else:
        print(f"Using single GPU")
        model.to(torch.device(device_name))
        criterion.to(torch.device(device_name))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, eps=1e-6)

    train_loss = []
    start_epoch = 1
    best_train_loss = float("inf")

    for epoch in range(start_epoch, cfg.epochs + 1):
        print(f"#####-------------- Epoch {epoch} started --------------#####")
        start_time = time.time()

        train_loss_epoch = train_epoch(model=model,
                                        train_dataloader=train_dataloader,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        use_cuda=use_cuda,
                                        writer=writer,
                                        epoch=epoch)

        train_loss.append(train_loss_epoch)
        writer.add_scalar("train/loss_epoch", train_loss_epoch, epoch)

        save_dir = os.path.join(cfg.saved_models_dir, cfg.run_id)

        if train_loss_epoch < best_train_loss:
            best_train_loss = train_loss_epoch
            print("++++++++++++++++++++++++++++++")
            print(f"Epoch {epoch} has the best model till now with Train Loss: {best_train_loss:.6f}")
            print("++++++++++++++++++++++++++++++")

            best_model_path = os.path.join(
                save_dir, f"model_epoch_{epoch}_loss_{best_train_loss:.6f}.pth"
            )
            save_checkpoint(model, optimizer, epoch, train_loss_epoch, best_model_path)

        temp_model_path = os.path.join(save_dir, "model_temp.pth")
        save_checkpoint(model, optimizer, epoch, train_loss_epoch, temp_model_path)

        losses_file = os.path.join(cfg.logs, cfg.run_id, "train_loss.txt")
        with open(losses_file, "w") as f:
            for loss in train_loss:
                f.write(f"{loss}\n")

        plot_file = os.path.join(cfg.logs, cfg.run_id, "train_loss_plot.png")
        plot_loss(train_loss, plot_file)

        taken = time.time() - start_time
        print(f"Time taken for Epoch-{epoch} is {taken:.2f} seconds")

    writer.close()
    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=[0],
        help="List of GPU device ids, e.g. --devices 0 or --devices 0 1"
    )
    args = parser.parse_args()

    train(args.devices)

