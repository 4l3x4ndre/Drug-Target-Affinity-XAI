# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from MGraphDTA.regression.metrics import get_cindex
from MGraphDTA.regression.dataset import *
from MGraphDTA.regression.model import MGraphDTA
from MGraphDTA.regression.utils import *
from MGraphDTA.regression.log.train_logger import TrainLogger

import wandb

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss

def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--dataset_path', required=True, type=str, help='path to dataset')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0')
    parser.add_argument('--model_path', default='', type=str, help='model path ready to load (to .pt file)')
    parser.add_argument('--wandb_log', action='store_true', help='use wandb to log training info')
    parser.add_argument('--job_id', type=str, default='test', help='job id for wandb')
    args = parser.parse_args()

    params = dict(
        save_dir="save",
        dataset_path=args.dataset_path,
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        wandb_log=args.wandb_log,
    )

    if params.get("wandb_log"):
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)
        wandb.init(project=os.getenv("WANDB_PROJECT"), config=params, name=f"reg-{args.dataset}-{args.job_id}")
    wandblog = params.get("wandb_log")

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    fpath = os.path.join(params.get('dataset_path'), DATASET)

    train_set = GNNDataset(fpath, train=True)
    test_set = GNNDataset(fpath, train=False)

    logger.debug(f'len(train_set)={len(train_set)} ; len(test_set)={len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=8)

    device = torch.device(params.get('device'))
    model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
    model_path = args.model_path
    if model_path != '':
        load_model_dict(model, model_path)

    epochs = 3000
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 400

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()

    logger.info('Strat training...')

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:

            global_step += 1       
            data = data.to(device)
            pred = model(data)

            loss = criterion(pred.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0)) 
            running_cindex.update(cindex, data.y.size(0))

            if global_step % steps_per_epoch == 0:

                global_epoch += 1

                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()

                test_loss = val(model, criterion, test_loader, device)

                msg = "epoch-%d, loss-%.4f, cindex-%.4f, test_loss-%.4f" % (global_epoch, epoch_loss, epoch_cindex, test_loss)
                logger.info(msg)

                if wandblog:
                    wandb.log({
                        "train/loss": epoch_loss,
                        "train/cindex": epoch_cindex,
                        "test/loss": test_loss,
                        "epoch": global_epoch,
                        "step": global_step
                    })

                if test_loss < running_best_mse.get_best():
                    running_best_mse.update(test_loss)
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), msg)
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break

if __name__ == "__main__":
    main()
