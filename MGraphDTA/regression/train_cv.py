import os
from dotenv import load_dotenv
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch_geometric.data import DataLoader
import wandb

from MGraphDTA.regression.preprocessing import GNNDatasetFull   # the unified dataset we wrote earlier
from MGraphDTA.regression.model import MGraphDTA
from MGraphDTA.regression.utils import AverageMeter, BestMeter, save_model_dict, load_model_dict
from MGraphDTA.regression.metrics import get_cindex, get_rm2
from MGraphDTA.regression.log.train_logger import TrainLogger

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    preds, labels = [], []

    with torch.no_grad():
        for data in tqdm(dataloader):
            data = data.to(device)
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            running_loss.update(loss.item(), data.y.size(0))
            preds.append(pred.view(-1).cpu().numpy())
            labels.append(data.y.cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    logger.debug(f'Validation predictions and labels collected. Total samples: {len(labels)}')
    rl = running_loss.get_average()
    logger.debug(f'Validation Loss: {rl:.4f}')
    ci = get_cindex(labels, preds)
    logger.debug(f'Validation C-Index: {ci:.4f}')
    r2 = get_rm2(labels, preds)
    logger.debug(f'Validation R2: {r2:.4f}')
    return rl, ci, r2


def train_fold(train_loader, val_loader, device, fold, rep, save_dir=None, wandblog=False):
    model = MGraphDTA(3, 26, embedding_size=128, filter_num=32, out_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    epochs = 3000
    steps_per_epoch = 50
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    early_stop_epoch = 400

    running_loss = AverageMeter()
    running_cindex = AverageMeter()
    running_best_mse = BestMeter("min")

    global_step, global_epoch = 0, 0
    break_flag = False

    logger.info('Start training...')

    for i in range(num_iter):
        if break_flag:
            break

        for data in tqdm(train_loader, desc=f"Training iteration {i+1}/{num_iter}"):
            global_step += 1
            data = data.to(device)
            pred = model(data)

            loss = criterion(pred.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.cpu().numpy().reshape(-1),
                                pred.detach().cpu().numpy().reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss.update(loss.item(), data.y.size(0))
            running_cindex.update(cindex, data.y.size(0))

            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                # reset and log training stats
                epoch_loss = running_loss.get_average()
                epoch_cindex = running_cindex.get_average()
                running_loss.reset()
                running_cindex.reset()
                logger.debug(f'Epoch {global_epoch}, step {global_step}: Train Loss: {epoch_loss:.4f}, Train C-Index: {epoch_cindex:.4f}')

                # validate
                val_loss, val_cindex, val_r2 = val(model, criterion, val_loader, device)
                logger.debug(f'Validation done. Val Loss: {val_loss:.4f}, Val C-Index: {val_cindex:.4f}, Val R2: {val_r2:.4f}')


                msg = (f"fold-{fold}, repeat-{rep}, "
                      f"epoch-{global_epoch}, train_loss-{epoch_loss:.4f}, "
                      f"train_cindex-{epoch_cindex:.4f}, "
                      f"val_loss-{val_loss:.4f}, val_cindex-{val_cindex:.4f}, val_r2-{val_r2:.4f}")
                logger.info(msg)

                if wandblog:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/running_loss_avg":epoch_loss,
                        "train/cindex": cindex,
                        "train/running_cindex_avg": epoch_cindex,
                        "train/epoch": global_epoch,
                        "train/step": global_step,
                        "val/loss": val_loss,
                        "val/cindex": val_cindex,
                        "val/r2": val_r2,
                        "fold": fold,
                        "repeat": rep,
                        "iteration": i+1
                    }, step=global_step)

                if val_loss < running_best_mse.get_best():
                    running_best_mse.update(val_loss)
                    if save_dir:
                        save_model_dict(model, logger.get_model_dir(), msg)
                else:
                    if running_best_mse.counter() > early_stop_epoch:
                        print(f"Early stop at epoch {global_epoch}")
                        break_flag = True
                        break

    return running_best_mse.get_best()


def generate_and_save_folds(dataset, k, repeats, file_path, logger):
    """
    Generates and saves cross-validation folds to a file.
    """
    logger.info(f"Generating {k}-fold cross-validation splits for {repeats} repeats...")
    all_folds = {}
    for rep in range(repeats):
        kf = KFold(n_splits=k, shuffle=True, random_state=rep)
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            all_folds[f'repeat_{rep}_fold_{fold}_train_idx'] = train_idx
            all_folds[f'repeat_{rep}_fold_{fold}_val_idx'] = val_idx
    np.savez(file_path, **all_folds)
    logger.info(f"Folds saved to {file_path}")

def run_single_fold_train(dataset, folds_file, fold, rep, device, params, logger):
    """
    Trains a single fold of a specific repeat from pre-generated fold indices.
    """
    logger.info(f"Loading folds from {folds_file} for repeat {rep}, fold {fold}")
    try:
        folds = np.load(folds_file)
        train_idx = folds[f'repeat_{rep}_fold_{fold}_train_idx']
        val_idx = folds[f'repeat_{rep}_fold_{fold}_val_idx']
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Could not load folds. Ensure you have generated them first. Error: {e}")
        return

    train_subset = dataset[train_idx.tolist()]
    val_subset = dataset[val_idx.tolist()]

    batch_size = params.get("batch_size", 256)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    logger.info(f"\n=== Training Repeat {rep}, Fold {fold} ===")
    best_val_loss = train_fold(train_loader, val_loader, device, fold, rep, params.get("save_dir"), params.get("wandb_log"))
    logger.info(f"Best validation loss for fold {fold}, repeat {rep}: {best_val_loss}")

    result_dir = "results/training/cv_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = os.path.join(result_dir, f"result_rep_{rep}_fold_{fold}.txt")
    with open(result_file, 'w') as f:
        f.write(str(best_val_loss))
    logger.info(f"Result saved to {result_file}")


def cross_validate(dataset, k=5, repeats=3, device="cuda:0", params=None):
    results = []
    params = params or {}
    batch_size = params.get('batch_size', 256)
    save_dir = params.get('save_dir')
    wandblog = params.get('wandb_log', False)

    for rep in range(repeats):
        kf = KFold(n_splits=k, shuffle=True, random_state=rep)
        logger.debug(f"=== Repeat {rep+1}/{repeats} ===")
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            logger.debug(f"=== Fold {fold+1}/{k} ===")
            train_subset = dataset[train_idx.tolist()]
            val_subset = dataset[val_idx.tolist()]

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size)

            logger.info(f"\n=== Repeat {rep+1}, Fold {fold+1}/{k} ===")
            best_val_loss = train_fold(train_loader, val_loader, device, fold, rep, save_dir, wandblog)
            results.append(best_val_loss)

    return results


if __name__ == "__main__":

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--dataset_path', required=True, type=str, help='path to dataset')
    parser.add_argument('--save_model_dir', type=str, default='models', help='directory to save models')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0')
    parser.add_argument('--wandb_log', action='store_true', help='use wandb to log training info')
    
    # New arguments for parallel execution
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for cross-validation')
    parser.add_argument('--repeats', type=int, default=3, help='number of repeats for cross-validation')
    parser.add_argument('--generate_folds', action='store_true', help='generate and save cross-validation folds')
    parser.add_argument('--folds_file', type=str, default='folds.npz', help='path to save/load folds')
    parser.add_argument('--fold', type=int, help='fold number to train')
    parser.add_argument('--repeat', type=int, help='repeat number to train')
    
    args = parser.parse_args()

    params = dict(
        data_root="data",
        dataset_path=args.dataset_path,
        save_dir=args.save_model_dir,
        dataset=args.dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        wandb_log=args.wandb_log,
        k_folds=args.k_folds,
        repeats=args.repeats,
        folds_file=args.folds_file,
        fold=args.fold,
        repeat=args.repeat,
    )
    

    if params.get("wandb_log"):
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)
        wandb.init(
            project=os.getenv("WANDB_PROJECT"), 
            config=params,
            tags=["regression", "cross-validation",
                  args.dataset],
        )

    logger = TrainLogger(params)
    logger.info(__file__)


    DATASET = params.get("dataset")
    fpath = os.path.join(params.get('dataset_path'), DATASET)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = GNNDatasetFull(fpath)

    if args.generate_folds:
        generate_and_save_folds(dataset, args.k_folds, args.repeats, args.folds_file, logger)
    elif args.fold is not None and args.repeat is not None:
        run_single_fold_train(dataset, args.folds_file, args.fold, args.repeat, device, params, logger)
    else:
        results = cross_validate(dataset, k=args.k_folds, repeats=args.repeats, device=device, params=params)
        print("CV results:", results)
        print("Mean val loss:", np.mean(results))

