from modules.RGCN import *
from modules.EdgeScores import *

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, agg_type='sum', drop=0.5):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names, agg_type, drop)
    def forward(self, g, x):
        _,output = self.sage(g, x)
        return output

class LinkPredictModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, agg_type='sum', drop=0.5):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names, agg_type, drop)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        hidden,_ = self.sage(g, x)
        return self.pred(g, hidden, etype), self.pred(neg_g, hidden, etype)

class MultiLink(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, agg_type='sum', drop=0.5, dot=True):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names, agg_type, drop)
        if dot:
            self.pred = HeteroDotProductPredictor()
        else:
            self.pred = MLPPredictor(hidden_features,1)
    def forward(self, g, neg_g, x, etype):
        hidden,output = self.sage(g, x)
        return self.pred(g, hidden, etype), self.pred(neg_g, hidden, etype),output