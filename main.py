import numpy as np
import random
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import argparse
from dataclasses import dataclass
import torch
# import wandb
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from deepptcl.dataset import ComboDataset
from deepptcl.models import BasedModel
from deepptcl.utils import (calculate_auprc, calculate_roc_auc, get_datasets, get_mol_dict)

WANDB_PROJECT = "PTCL"

# Fixed model parameters
NUM_LAYERS = 5
INJECT_LAYER = 3
EMB_DIM = 300
FEATURE_DIM = 512
BATCH_SIZE = 64
LR = 5e-4


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


@dataclass
class TrainConfiguration:
    synergy_score: str
    fold_number: int
    number_of_epochs: int
    data_folder_path: str
    test_mode: bool
    ML_output: str


def evaluate(model, loader, loss_fn, device, test_mode, regression=False):
    model.eval()
    epoch_preds, epoch_labels, epoch_loss = [], [], 0.0

    for batch in loader:
        drugA, drugB, cell_line, target = [tensor.to(device) for tensor in batch]
        with torch.no_grad():
            output = model(drugA, drugB, cell_line)
        loss = loss_fn(output, target)
        epoch_preds.append(output.cpu())
        epoch_labels.append(target.cpu())
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    epoch_preds, epoch_labels = torch.cat(epoch_preds), torch.cat(epoch_labels)

    if test_mode:
        pd.DataFrame({'preds': epoch_preds.reshape(-1).tolist()}).to_csv('outputs/preds_ptcl_CellTPM.csv')

    if regression:
        prc = pearsonr(epoch_labels.flatten(), epoch_preds.flatten()).statistic
        spc = spearmanr(epoch_labels.flatten(), epoch_preds.flatten()).statistic
        if wandb.run:
            wandb.log({"val_pearsonr": prc, "val_spearmanr": spc, "val_loss": epoch_loss})
        return prc
    else:
        auprc = calculate_auprc(epoch_labels, epoch_preds)
        auc = calculate_roc_auc(epoch_labels, epoch_preds)
        if wandb.run:
            wandb.log({"val_auprc": auprc, "val_auc": auc, "val_loss": epoch_loss})
        return auc


def train_model(model, config, device, regression=False):
    dataset, train_data, test_data, cell_lines = get_datasets(config.data_folder_path, config.fold_number,
                                                              config.synergy_score, False, None, config.test_mode)
    mol_mapping = get_mol_dict(dataset)

    train_set = ComboDataset(train_data, cell_lines, mol_mapping)
    test_set = ComboDataset(test_data, cell_lines, mol_mapping)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss() if regression else nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(config.number_of_epochs)):
        model.train()
        for batch in train_loader:
            drugA, drugB, cell_line, target = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            output = model(drugA, drugB, cell_line)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            test_metric = evaluate(model, test_loader, loss_fn, device, config.test_mode, regression)
            torch.save(model.state_dict(), f'weights/model_{config.ML_output}.pth')


def test_model(model, config, device):
    dataset, _, test_data, cell_lines = get_datasets(config.data_folder_path, config.fold_number, config.synergy_score,
                                                     False, None, config.test_mode)
    mol_mapping = get_mol_dict(dataset)

    test_set = ComboDataset(test_data, cell_lines, mol_mapping)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

    model.load_state_dict(torch.load(f'weights/model_{config.ML_output}.pth', map_location=torch.device('cpu')))
    model.to(device)
    loss_fn = nn.MSELoss() if config.ML_output == 'regression' else nn.BCEWithLogitsLoss()

    evaluate(model, test_loader, loss_fn, device, config.test_mode, regression=config.ML_output == 'regression')


def train(config):
    set_seed()
    if config.with_wandb:
        wandb.init(config=config, project=WANDB_PROJECT)
        print(f'Hyperparameters:\n {wandb.config}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BasedModel(
        num_layers=NUM_LAYERS,
        inject_layer=INJECT_LAYER,
        emb_dim=EMB_DIM,
        feature_dim=FEATURE_DIM,
        device=device
    )

    train_config = TrainConfiguration(
        synergy_score=config.synergy_score,
        fold_number=config.fold_number,
        number_of_epochs=config.number_of_epochs,
        data_folder_path=config.data_folder_path,
        test_mode=config.test_mode,
        ML_output=config.ML_output
    )

    if config.test_mode:
        test_model(model, train_config, device)
    else:
        train_model(model, train_config, device, regression=config.ML_output == "regression")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a deepptcl model')
    parser.add_argument('--synergy_score', type=str, default="loewe")
    parser.add_argument('--fold_number', type=int, default=4)
    parser.add_argument('--number_of_epochs', type=int, default=50)
    parser.add_argument('--data_folder_path', type=str, default="data/")
    parser.add_argument('--with_wandb', type=bool, default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--ML_output', type=str, default='loewe-class')

    config = parser.parse_args()
    train(config)
