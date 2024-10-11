import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool
from typing import List, Tuple

class BasedModel(nn.Module):
    """Graph-based model utilizing GraphProcessor, MLP, and Layers."""

    def __init__(self,
                 num_layers: int = 5,
                 inject_layer: int = 3,
                 emb_dim: int = 300,
                 mlp_hidden_dims: List[int] = [256, 128, 64],
                 feature_dim: int = 512,
                 device=torch.device('cuda')
                 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.device = device

        # Embedding layers for atom and chirality types
        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        # Generate basic layers (before inject_layer)
        basic_layers_number = inject_layer
        ptcl_layers_number = num_layers - inject_layer
        self.basic_layers = self._generate_basic_layers(basic_layers_number)
        self.ptcl_layers = self._generate_ptcl_layers(ptcl_layers_number)

        # Output MLP
        self.output_transformation = MLP(emb_dim, [feature_dim], feature_dim // 2, activation="relu")
        self.mlp = MLP(self.emb_dim + feature_dim // 2, mlp_hidden_dims, 1, activation="leaky_relu")

    def _generate_basic_layers(self, number_of_layers: int) -> nn.ModuleList:
        """Generates basic layers using GraphProcessor."""
        basic_layers = []
        for i in range(number_of_layers):
            graph_processor = GraphProcessor(self.emb_dim, self.emb_dim, NUM_ATOM_TYPE, NUM_CHIRALITY_TAG).to(self.device)
            basic_layers.append(graph_processor)
        return nn.ModuleList(basic_layers)

    def _generate_ptcl_layers(self, number_of_layers: int) -> nn.ModuleList:
        """Generates layers after the inject point using Layers."""
        ptcl_layers = []
        for i in range(number_of_layers):
            layer = Layers(self.emb_dim, self.emb_dim, NUM_ATOM_TYPE, NUM_CHIRALITY_TAG).to(self.device)
            ptcl_layers.append(layer)
        return nn.ModuleList(ptcl_layers)

    def _embed_x(self, graph: Data) -> Data:
        """Embeds atom and chirality features into the graph."""
        embedding_1 = self.x_embedding1(graph.x[:, 0])
        embedding_2 = self.x_embedding2(graph.x[:, 1])
        graph.x = embedding_1 + embedding_2
        return graph

    def forward(self, graphA: Data, graphB: Data) -> torch.Tensor:
        """Forward pass for processing two graphs."""
        # Embed node features for both graphs
        graphA = self._embed_x(graphA)
        graphB = self._embed_x(graphB)

        # Apply basic layers
        for layer in self.basic_layers:
            graphA, graphB = layer(graphA, graphB)

        # Apply Ptcl layers
        for layer in self.ptcl_layers:
            graphA, graphB = layer(graphA, graphB)

        # Global mean pooling
        graphA.x = global_mean_pool(graphA.x, graphA.batch)
        graphA.x = self.output_transformation(graphA.x)

        graphB.x = global_mean_pool(graphB.x, graphB.batch)
        graphB.x = self.output_transformation(graphB.x)

        # Concatenate outputs from both graphs and apply final MLP
        input_ = torch.cat((graphA.x, graphB.x), dim=1)
        return self.mlp(input_)


class GraphProcessor(MessagePassing):
    """Processes graph node features and handles graph updates."""

    def __init__(self, node_dim: int, out_dim: int, edge_type_dim: int, edge_dir_dim: int, use_relu: bool = True) -> None:
        super().__init__()
        # MLP for node updates
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, 2 * node_dim),
            nn.ReLU(),
            nn.Linear(2 * node_dim, out_dim)
        )
        # Edge embeddings
        self.edge_embed_type = nn.Embedding(edge_type_dim, node_dim)
        self.edge_embed_dir = nn.Embedding(edge_dir_dim, node_dim)
        nn.init.xavier_uniform_(self.edge_embed_type.weight)
        nn.init.xavier_uniform_(self.edge_embed_dir.weight)
        # Normalization and optional ReLU
        self.norm = nn.BatchNorm1d(out_dim)
        self.use_relu = use_relu

    def forward(self, graphA: Data, graphB: Data) -> Tuple[Data, Data]:
        graphA.x = self._process_graph(graphA.x, graphA.edge_index, graphA.edge_attr)
        graphB.x = self._process_graph(graphB.x, graphB.edge_index, graphB.edge_attr)
        return graphA, graphB

    def _process_graph(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Add self-loops to the edge index
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        # Create self-loop edge attributes
        self_loop_attr = torch.zeros(x.size(0), 2, device=edge_attr.device, dtype=edge_attr.dtype)
        self_loop_attr[:, 0] = 4  # Self-loop bond type
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        # Compute edge embeddings
        edge_emb = self.edge_embed_type(edge_attr[:, 0])
        # Propagate the graph data
        x = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        # Apply batch normalization and optional ReLU
        x = self.norm(x)
        if self.use_relu:
            x = F.relu(x)
        return x

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return self.node_mlp(aggr_out)


class MLP(nn.Module):
    """Multi-layer perceptron with flexible depth and activation functions."""

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, activation: str = "relu", dropout: float = 0.0, slope: float = 0.01) -> None:
        super().__init__()
        layers = []
        # Add first layer
        layers.append(self._create_layer(input_dim, hidden_dims[0], activation, dropout, slope))
        # Add intermediate layers
        for h_in, h_out in zip(hidden_dims, hidden_dims[1:]):
            layers.append(self._create_layer(h_in, h_out, activation, dropout, slope))
        # Add final layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)
        self._initialize_weights(activation, slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def _create_layer(self, in_dim: int, out_dim: int, activation: str, dropout: float, slope: float) -> nn.Module:
        if activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.LeakyReLU(slope)
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            act,
            nn.Dropout(dropout)
        )

    def _initialize_weights(self, activation: str, slope: float) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=slope, nonlinearity=activation)
                nn.init.uniform_(layer.bias, -1, 0)
        nn.init.xavier_normal_(self.mlp[-1].weight)
        nn.init.uniform_(self.mlp[-1].bias, -1, 0)


class Layers(nn.Module):
    """Layer combining graph updates and context combination."""

    def __init__(self, node_dim: int, out_dim: int, edge_type_dim: int, edge_dir_dim: int, use_relu: bool = True) -> None:
        super().__init__()
        self.processor = GraphProcessor(node_dim, out_dim, edge_type_dim, edge_dir_dim, use_relu)

    def forward(self, graphA: Data, graphB: Data) -> Tuple[Data, Data, torch.Tensor]:
        graphA, graphB = self.processor(graphA, graphB)
        # Combine the node features from both graphs and context
        return graphA, graphB
