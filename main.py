import argparse
from dataclasses import dataclass

import torch
import wandb
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from congfu.dataset import DrugCombDataset
from congfu.models import CongFuBasedModel
from congfu.utils import (calculate_auprc, calculate_roc_auc, get_datasets,
                          get_mol_dict)
import numpy as np
import random
import os
import pandas as pd
from scipy.stats import pearsonr, spearmanr

WANDB_PROJECT="DLBCL"
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

@dataclass
class TrainConfiguration:
    synergy_score: str
    transductive: bool
    inductive_set_name: str
    fold_number: int
    batch_size: int
    lr: float
    number_of_epochs: int
    data_folder_path: str
    test_mode: bool
    ML_output:str

def evaluate_mlp(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device,test_mode) -> None:
    model.eval()

    epoch_preds, epoch_labels = [], []
    epoch_loss = 0.0

    drugAL, drugBL, cell_lineL = [],[],[]

    for batch in loader:
        batch = [tensor.to(device) for tensor in batch]
        drugA, drugB, cell_line, target = batch

        with torch.no_grad():
            output = model(drugA, drugB, cell_line)

        loss = loss_fn(output, target)
        epoch_preds.append(output.detach().cpu())
        epoch_labels.append(target.detach().cpu())
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)


    if test_mode == True:
        pd.DataFrame({'preds': epoch_preds.detach().cpu().reshape(-1).tolist()}).to_csv(
            'outputs/preds_DLBCL.csv')

    auprc = calculate_auprc(epoch_labels, epoch_preds)
    auc = calculate_roc_auc(epoch_labels, epoch_preds)
    
    if wandb.run is not None:
        wandb.log({"val_auprc": auprc, "val_auc": auc, "val_loss": epoch_loss})

    print(auc,auprc)


    return auc


def evaluate_mlp_regression(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device, test_mode) -> None:
    model.eval()

    epoch_preds, epoch_labels = [], []
    epoch_loss = 0.0

    drugAL, drugBL, cell_lineL = [], [], []

    for batch in loader:
        batch = [tensor.to(device) for tensor in batch]
        drugA, drugB, cell_line, target = batch

        with torch.no_grad():
            output = model(drugA, drugB, cell_line)

        loss = loss_fn(output, target)
        epoch_preds.append(output.detach().cpu())
        epoch_labels.append(target.detach().cpu())
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    epoch_preds = torch.cat(epoch_preds)
    epoch_labels = torch.cat(epoch_labels)

    if test_mode == True:
        pd.DataFrame({'preds': epoch_preds.detach().cpu().reshape(-1).tolist()}).to_csv(
            'outputs/preds_DLBCL.csv')

    prc = pearsonr(epoch_labels.detach().cpu().reshape(-1).tolist(), epoch_preds.detach().cpu().reshape(-1).tolist()).statistic
    spc = spearmanr(epoch_labels.detach().cpu().reshape(-1).tolist(), epoch_preds.detach().cpu().reshape(-1).tolist()).statistic

    if wandb.run is not None:
        wandb.log({"val_pearsonr": prc, "val_spearmanr": spc, "val_loss": epoch_loss})


    return prc

def train_model(model: nn.Module, config: TrainConfiguration, device: torch.device) -> None:
    dataset, train_dataset, test_dataset, cell_lines = get_datasets(config.data_folder_path, config.fold_number, config.synergy_score, config.transductive, config.inductive_set_name, config.test_mode)
    mol_mapping = get_mol_dict(dataset)

    train_set = DrugCombDataset(train_dataset, cell_lines, mol_mapping)
    test_set = DrugCombDataset(test_dataset, cell_lines, mol_mapping)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=4, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_fn = nn.BCEWithLogitsLoss()

    model.train()

    test_auc_objects = []
    test_auc_objects.append(0)

    for _ in tqdm(range(config.number_of_epochs)):
        epoch_preds, epoch_labels = [], []
        epoch_loss = 0.0

        for batch in train_loader:
            batch = [tensor.to(device) for tensor in batch]
            drugA, drugB, cell_line, target = batch

            optimizer.zero_grad()
            output = model(drugA, drugB, cell_line)
            loss = loss_fn(output, target)

            epoch_preds.append(output.detach().cpu())
            epoch_labels.append(target.detach().cpu())
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_loader)
        epoch_preds = torch.cat(epoch_preds)
        epoch_labels = torch.cat(epoch_labels)

        auprc = calculate_auprc(epoch_labels, epoch_preds)
        auc = calculate_roc_auc(epoch_labels, epoch_preds)

        if wandb.run is not None:
            wandb.log({"train_auprc": auprc, "train_auc": auc, "train_loss": epoch_loss})


        # if _ % 10 == 0:
        #     # if test_auc >= test_auc_objects[-1]:
        #     torch.save(model.state_dict(), 'weights/model_DLBCL.pth')
        # test_auc_objects.append(test_auc)
        torch.save(model.state_dict(), 'weights/model_DLBCL.pth')

        model.load_state_dict(torch.load('weights/model_DLBCL.pth', map_location=torch.device('cpu')))
        model.to(device)
        loss_fn = nn.BCEWithLogitsLoss()

        all_set = DrugCombDataset(dataset, cell_lines, mol_mapping)
        test_loader = DataLoader(all_set, batch_size=config.batch_size, num_workers=4, shuffle=False)
        test_auc = evaluate_mlp(model, test_loader, loss_fn, device, config.test_mode)
        print("Test set AUC: {:.4f}".format(test_auc))


def test_giogrio(model: nn.Module, config: TrainConfiguration, device: torch.device) -> None:
    dataset, train_dataset, test_dataset, cell_lines = get_datasets(config.data_folder_path, config.fold_number, config.synergy_score, config.transductive, config.inductive_set_name,\
                                                                    config.test_mode)
    mol_mapping = get_mol_dict(dataset)

    test_set = DrugCombDataset(dataset, cell_lines, mol_mapping)

    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=4, shuffle=False)

    model.load_state_dict(torch.load('weights/model_DLBCL.pth',map_location=torch.device('cpu')))
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    loss_fn_regr = nn.MSELoss()

    if config.ML_output != 'regression':
        test_auc = evaluate_mlp(model, test_loader, loss_fn, device, config.test_mode)

    else:
        test_pearson = evaluate_mlp_regression(model, test_loader, loss_fn_regr, device, config.test_mode)

def train_model_regression(model: nn.Module, config: TrainConfiguration, device: torch.device) -> None:
    dataset, train_dataset, test_dataset, cell_lines = get_datasets(config.data_folder_path, config.fold_number, config.synergy_score, config.transductive, config.inductive_set_name, config.test_mode)
    mol_mapping = get_mol_dict(dataset)

    train_set = DrugCombDataset(train_dataset, cell_lines, mol_mapping)
    test_set = DrugCombDataset(test_dataset, cell_lines, mol_mapping)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, num_workers=4, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_fn = nn.MSELoss()

    model.train()

    test_pearson_objects = []
    test_pearson_objects.append(0)

    for _ in tqdm(range(config.number_of_epochs)):
        epoch_preds, epoch_labels = [], []
        epoch_loss = 0.0

        for batch in train_loader:
            batch = [tensor.to(device) for tensor in batch]
            drugA, drugB, cell_line, target = batch

            optimizer.zero_grad()
            output = model(drugA, drugB, cell_line)
            loss = loss_fn(output, target)

            epoch_preds.append(output.detach().cpu())
            epoch_labels.append(target.detach().cpu())
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / len(train_loader)
        epoch_preds = torch.cat(epoch_preds)
        epoch_labels = torch.cat(epoch_labels)

        prc = pearsonr(epoch_labels.detach().cpu().reshape(-1).tolist(),
                       epoch_preds.detach().cpu().reshape(-1).tolist()).statistic
        spc = spearmanr(epoch_labels.detach().cpu().reshape(-1).tolist(),
                        epoch_preds.detach().cpu().reshape(-1).tolist()).statistic

        if wandb.run is not None:
            wandb.log({"train_pearsonr": prc, "train_spearmanr": spc, "train_loss": epoch_loss})


        test_pearson = evaluate_mlp_regression(model, test_loader, loss_fn, device, TrainConfiguration)

        if _ % 10 == 0:
            if test_pearson >= test_pearson_objects[-1]:
                torch.save(model.state_dict(), 'weights/model_regr.pth')
        test_pearson_objects.append(test_pearson)

def train(config):
    set_seed()
    if config.with_wandb:
        wandb.init(config=config, project=WANDB_PROJECT)
        print(f'Hyper parameters:\n {wandb.config}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CongFuBasedModel(
        num_layers=config.num_layers,
        inject_layer = config.inject_layer,
        emb_dim = config.emb_dim,
        feature_dim = config.feature_dim,
        context_dim = config.context_dim,
        device=device
    )

    train_configuraion = TrainConfiguration(
        synergy_score=config.synergy_score,
        transductive = config.transductive,
        inductive_set_name = config.inductive_set_name,
        lr = config.lr,
        number_of_epochs = config.number_of_epochs,
        data_folder_path=config.data_folder_path,
        fold_number = config.fold_number,
        batch_size=config.batch_size,
        test_mode=config.test_mode,
        ML_output=config.ML_output
    )
    if not config.test_mode:

        if config.ML_output == "regression":
            train_model_regression(model, train_configuraion, device)
        else:
            train_model(model, train_configuraion, device)

    else:
        test_giogrio(model, train_configuraion, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CongFu-based model')
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--inject_layer', type=int, default=3)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--context_dim', type=int, default=883)#908
    parser.add_argument('--synergy_score', type=str, default="astrazeneca")#loewe, giorgio,
    parser.add_argument('--transductive', type=bool, default=False)
    parser.add_argument('--inductive_set_name', type=str, default="leave_combo")
    parser.add_argument('--fold_number', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--number_of_epochs', type=int, default=50)
    parser.add_argument('--data_folder_path', type=str, default="data/preprocessed/")
    parser.add_argument('--with_wandb', type=bool,default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--ML_output', type=str, default='classification')#regression

    config = parser.parse_args()
    train(config)
