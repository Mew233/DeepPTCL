import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from typing import Tuple, Dict
import json
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm

# Constants
ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


# -------------------- Data Processing Functions --------------------

def split_fold(dataset: pd.DataFrame, fold: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into training and testing folds based on provided fold indices."""
    train_indices, test_indices = fold["train"], fold["test"]
    X_train = dataset.iloc[train_indices]
    X_test = dataset.iloc[test_indices]
    return X_train, X_test


def get_datasets(data_folder_path: str,
                 fold_number: int,
                 synergy_score: str,
                 transductive: bool,
                 inductive_set_name: str,
                 test_mode: bool
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepares and returns datasets for training and testing based on the evaluation setup and synergy score.

    Args:
    - data_folder_path: str: Path to the data folder.
    - fold_number: int: Fold number for the data split.
    - synergy_score: str: The synergy score being used.
    - transductive: bool: Whether the evaluation is transductive or inductive.
    - inductive_set_name: str: Name of the inductive set.
    - test_mode: bool: Whether the model is in test mode.

    Returns:
    - dataset: Full dataset (concatenated training and testing sets).
    - train_dataset: Training dataset.
    - test_dataset: Testing dataset.
    - cell_lines: Cell line embeddings.
    """
    cell_lines = pd.read_feather(data_folder_path + f"{synergy_score}.feather").set_index("cell_line_name")
    cell_lines = cell_lines.astype(np.float32)

    if transductive:
        dataset = pd.read_feather(data_folder_path + f"{synergy_score}/{synergy_score}.feather")
        with open(data_folder_path + f"{synergy_score}/{synergy_score}.json") as f:
            folds = json.load(f)
        fold = folds[f"fold_{fold_number}"]
        train_dataset, test_dataset = split_fold(dataset, fold)
    else:
        train_dataset = pd.read_feather(
            data_folder_path + f"{synergy_score}/{inductive_set_name}/train_{fold_number}.feather")
        test_dataset = pd.read_feather(
            data_folder_path + f"{synergy_score}/{inductive_set_name}/test_{fold_number}.feather")
        dataset = pd.concat([train_dataset, test_dataset])

    return dataset, train_dataset, test_dataset, cell_lines


# -------------------- Metrics Calculation Functions --------------------

def calculate_roc_auc(targets: np.array, preds: np.array) -> float:
    """Calculates and returns the ROC AUC score."""
    return roc_auc_score(targets, preds)


def calculate_auprc(targets: np.array, preds: np.array) -> float:
    """Calculates and returns the AUPRC (Area Under Precision-Recall Curve)."""
    precision_scores, recall_scores, _ = precision_recall_curve(targets, preds)
    return auc(recall_scores, precision_scores)


# -------------------- Molecule Processing Functions --------------------

def _get_drug_tokens(smiles: str) -> Data:
    """
    Converts SMILES string to PyTorch Geometric Data format.

    Args:
    - smiles: str: SMILES representation of a molecule.

    Returns:
    - Data: PyTorch Geometric Data object with atom and bond information.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Extract atom and chirality features
    atom_features = [(ATOM_LIST.index(atom.GetAtomicNum()), CHIRALITY_LIST.index(atom.GetChiralTag())) for atom in
                     mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.long)

    # Extract bond features and adjacency information
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), BOND_LIST.index(bond.GetBondType()),
              BONDDIR_LIST.index(bond.GetBondDir())) for bond in mol.GetBonds()]
    edge_index = torch.tensor(
        [[start, end] for start, end, _, _ in bonds] + [[end, start] for start, end, _, _ in bonds],
        dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([[bt, bd] for _, _, bt, bd in bonds] + [[bt, bd] for _, _, bt, bd in bonds],
                             dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_mol_dict(df: pd.DataFrame) -> Dict:
    """
    Creates a dictionary mapping drug IDs to their molecular graphs (as PyTorch Geometric Data objects).

    Args:
    - df: pd.DataFrame: DataFrame containing 'Drug1_ID', 'Drug1', 'Drug2_ID', 'Drug2' columns.

    Returns:
    - dict: Dictionary where keys are drug IDs and values are molecular graphs (PyTorch Geometric Data).
    """
    drugs = pd.concat([
        df[['Drug1_ID', 'Drug1']].rename(columns={'Drug1_ID': 'id', 'Drug1': 'drug'}),
        df[['Drug2_ID', 'Drug2']].rename(columns={'Drug2_ID': 'id', 'Drug2': 'drug'})
    ]).drop_duplicates(subset='id')

    mol_dict = {row['id']: _get_drug_tokens(row['drug']) for _, row in tqdm(drugs.iterrows(), total=len(drugs))}
    return mol_dict
