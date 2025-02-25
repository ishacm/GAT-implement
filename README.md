## PyTorch implementation of the paper Graph Attention Networks


### Key Features:
- Multi-head self-attention mechanism
- Handles varying neighborhood sizes
- No requirement for pre-defined adjacency weights

### Installation
```bash
pip install torch labml
```

### Usage
```
from labml_nn.graphs.gat import GAT
model = GAT(in_features=1433, hidden_dim=8, out_features=7, num_heads=8)
```

### Reference
Paper: Graph Attention Networks\
LabML Implementation: nn.labml.ai
