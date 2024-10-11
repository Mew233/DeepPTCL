import torch
from torch.utils.data import Dataset

# Column names for drug and cell line identifiers
DRUG_A = "Drug1_ID"
DRUG_B = "Drug2_ID"
CELL_LINE = "Cell_Line_ID"


class ComboDataset(Dataset):
    """
    Custom Dataset for handling drug combination data with cell line embeddings.
    Prepares drug and cell line features along with the corresponding target values.
    """

    def __init__(self, data, cell_data, drug_map):
        """
        Initializes the dataset with combination data, cell line embeddings, and drug embeddings.

        Parameters:
        -----------
        data : pd.DataFrame
            Data containing the combination information for drugs and targets.

        cell_data : pd.DataFrame
            Embedding data for cell lines.

        drug_map : dict
            Dictionary mapping each drug to its embedding.
        """
        self.data = data
        self.cell_data = cell_data
        self.drug_map = drug_map
        self.labels = torch.tensor(data['target'].values, dtype=torch.float32)

    def __len__(self):
        """
        Returns the number of records in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single record by index.

        Parameters:
        -----------
        idx : int
            Index of the record.

        Returns:
        --------
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
            - Embeddings for drug A
            - Embeddings for drug B
            - Cell line embeddings
            - Target value (synergy score)
        """
        record = self.data.iloc[idx]

        # Get drug embeddings
        drug_a_emb = self.drug_map[record[DRUG_A]]
        drug_b_emb = self.drug_map[record[DRUG_B]]

        # Get cell line embeddings
        cell_emb = torch.tensor(self.cell_data.loc[record[CELL_LINE]].values, dtype=torch.float32)

        # Get target value
        target = self.labels[idx].unsqueeze(0)

        return drug_a_emb, drug_b_emb, cell_emb, target