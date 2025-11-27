# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import argparse
from tqdm import tqdm
import gc

from metrics import get_cindex, get_rm2
from dataset import *
from model import MGraphDTA
from utils import *

def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred = np.zeros(len(dataloader.dataset), dtype=np.float32)
    label = np.zeros(len(dataloader.dataset), dtype=np.float32)
    start_idx = 0

    for data in tqdm(dataloader):
        data = data.to(device)
        batch_size = data.y.size(0)

        with torch.no_grad():
            out = model(data)
            loss = criterion(out.view(-1), data.y.view(-1))
            # label = data.y
            # pred_list.append(out.view(-1).detach().cpu().numpy())
            # label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), data.y.size(0))

        pred[start_idx:start_idx + batch_size] = out.view(-1).cpu().numpy()
        label[start_idx:start_idx + batch_size] = data.y.cpu().numpy()
        start_idx += batch_size

    # pred = np.concatenate(pred_list, axis=0)
    # label = np.concatenate(label_list, axis=0)

    print("Computing cindex...")
    epoch_cindex = get_cindex(label, pred)
    print("Computing r2")
    epoch_r2 = get_rm2(label, pred)
    print("Computing loss")
    epoch_loss = running_loss.get_average()
    print("Reset running loss")
    running_loss.reset()
    print("return")

    return epoch_loss, epoch_cindex, epoch_r2

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def multi_test(model_path, dataset, device, seeds=[0, 1, 2, 3, 4]):
    data_root = "data"
    fpath = os.path.join(data_root, dataset)
    test_set = GNNDataset(fpath, train=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)

    results = []
    for seed in seeds:
        print(f"{seed}/{len(seeds)}")
        set_seed(seed)
        model = MGraphDTA(3, 26, embedding_size=128, filter_num=32, out_dim=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        criterion = nn.MSELoss()

        loss, cindex, r2 = val(model, criterion, test_loader, device)
        results.append((loss, cindex, r2))
        print(f"Seed {seed}: loss={loss:.4f}, cindex={cindex:.4f}, r2={r2:.4f}")

        del model, criterion
        gc.collect()

    results = np.array(results)
    print("\nAverage over seeds:")
    print(f"Loss: {results[:,0].mean():.4f} ± {results[:,0].std():.4f}")
    print(f"C-index: {results[:,1].mean():.4f} ± {results[:,1].std():.4f}")
    print(f"R²: {results[:,2].mean():.4f} ± {results[:,2].std():.4f}")


def main():
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--dataset', required=True, help='davis or kiba')
    parser.add_argument('--model_path', required=True, type=str, help='model path ready to load')
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:0')
    args = parser.parse_args()

    # data_root = "data"
    # DATASET = args.dataset
    # model_path = args.model_path
    # fpath = os.path.join(data_root, DATASET)
    # test_set = GNNDataset(fpath, train=False)
    # print("Number of test: ", len(test_set))
    # test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=8)
    # device = torch.device(args.device)
    # model = MGraphDTA(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=1).to(device)
    # criterion = nn.MSELoss()
    # load_model_dict(model, model_path)
    # test_loss, test_cindex, test_r2 = val(model, criterion, test_loader, device)
    # msg = "test_loss:%.4f, test_cindex:%.4f, test_r2:%.4f" % (test_loss, test_cindex, test_r2)
    # print(msg)

    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    args = parser.parse_args()

    multi_test(args.model_path, args.dataset, args.device, args.seeds)


if __name__ == "__main__":
    main()
