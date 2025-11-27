import torch.nn as nn
import copy
import torch.nn as nn
from torch_geometric.data import Batch

class FullModelForGradCAM(nn.Module):
    """
    Wrap the original full model so that GradCAM can call it with a tensor input.
    - `full_model` is your trained MGraphDTA (or MGraphDTAXAI).
    - `ligand_template` is a Data object (from test_set[idx]) that provides the ligand/graph fields;
       the wrapper will replace its `target` with the sequence passed to forward.
    """
    def __init__(self, full_model, ligand_template):
        super().__init__()
        self.full_model = full_model
        # keep a CPU copy of template; we'll move copies to the right device later
        self.ligand_template = ligand_template

    def forward(self, x):
        # x is expected to be either:
        #  - [B, 1, 1, L] or
        #  - [B, L]  (tokens)
        if x.dim() == 4:
            seq = x.squeeze(1).squeeze(2)   # -> [B, L]
        elif x.dim() == 2:
            seq = x
        else:
            raise ValueError(f"Unsupported input dim {x.dim()} for GradCAM wrapper")

        # Prepare a batch of Data objects (one per sample) by copying the ligand template
        datas = []
        device = next(self.full_model.parameters()).device

        for i in range(seq.size(0)):
            d = copy.deepcopy(self.ligand_template)  # micro-copy to change just this sample
            d.target = seq[i].to(d.target.dtype).to(device)
            datas.append(d)

        batch = Batch.from_data_list(datas).to(device)
        # Now call the original model with the constructed Batch
        out = self.full_model(batch) 
        if out.dim() == 2 and out.shape[1] == 1:
            return out.view(-1)
        return out

