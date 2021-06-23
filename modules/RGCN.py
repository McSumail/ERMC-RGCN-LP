import dgl.nn as dglnn
import torch.nn.functional as F
import torch.nn as nn

class RGCN(nn.Module):
   def __init__(self, in_feats, hid_feats, out_feats, rel_names, agg_type='sum', drop=0.5):
      super().__init__()

      self.conv1 = dglnn.HeteroGraphConv({
         rel: dglnn.GraphConv(in_feats, hid_feats)
         for rel in rel_names}, aggregate=agg_type)
      self.conv2 = dglnn.HeteroGraphConv({
         rel: dglnn.GraphConv(hid_feats, hid_feats)
         for rel in rel_names}, aggregate=agg_type)
      self.dropout=nn.Dropout(drop)
      self.Linear=nn.Linear(hid_feats, out_feats)

   def forward(self, graph, inputs, exlinear=True):
      hidden = self.conv1(graph, inputs)
      hidden = {k: F.relu(v) for k, v in hidden.items()}
      hidden = self.conv2(graph, hidden)
      hidden = {k: F.relu(v) for k, v in hidden.items()}
      h = {k: self.dropout(v) for k, v in hidden.items()}
      output = {k: self.Linear(v) for k, v in h.items()}

      return hidden,output



