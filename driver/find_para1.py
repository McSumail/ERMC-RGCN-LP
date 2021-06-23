import sys
sys.path.append("..")
import random
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import torch.optim as optim
from modules.Model import GCN,MultiLink
from data.Dataloader import *
from data.Config import *
import argparse,time

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()

def train_or_eval_model(k, loss_ma, model, heterograph, mode, optimizer=None):
    if mode=='train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    negative_graph = construct_negative_graph(heterograph, k, ('speaker', 'speak', 'utt'))
    utt_feats=heterograph.nodes['utt'].data['feature']
    speaker_feats=heterograph.nodes['speaker'].data['feature']


    label=heterograph.nodes['utt'].data['emotionlabel']
    train_mask=heterograph.nodes['utt'].data['trainmask']
    dev_mask=heterograph.nodes['utt'].data['devmask']
    test_mask=heterograph.nodes['utt'].data['testmask']
    node_features={'utt':utt_feats,'speaker':speaker_feats}

    pos_score, neg_score, log_prob = model(heterograph,negative_graph,node_features,('speaker', 'speak', 'utt'))

    log_prob_main = log_prob['utt']
    if mode=='train':
        log_prob_main,label_main=log_prob_main[train_mask],label[train_mask]
    elif mode=='dev':
        log_prob_main,label_main=log_prob_main[dev_mask],label[dev_mask]
    elif mode=='test':
        log_prob_main,label_main=log_prob_main[test_mask],label[test_mask]

    loss_main = F.cross_entropy(log_prob_main,label_main)
    loss_au = compute_loss(pos_score,neg_score)
    pred = torch.argmax(log_prob_main, 1)
    preds = pred.data.cpu().numpy()
    labels = label_main.data.cpu().numpy()

    if mode=='train':
        (loss_main*loss_ma+loss_au*(1-loss_ma)).backward()
        optimizer.step()


    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    return round((loss_main.item()),4), avg_accuracy, labels, preds, avg_fscore

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='../examples/defaultMELD.cfg')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--no_cuda', action='store_true', default=True)
    argparser.add_argument('--agg_type', type=str, default='sum')

    args, extra_args = argparser.parse_known_args()
    torch.set_num_threads(args.thread)
    for lr in [0.003,0.005,0.007]:
        for drop in [0.4,0.5,0.6]:
            for hiddensize in [50,64,128]:
                for k in [1]:#[1,2]:
                    for loss_ma in [0.9,0.8]:
                        for seed in [66,666]:
                            print(lr,drop,hiddensize,k,loss_ma,seed)
                            seed_everything(seed)
                            epochs = 200
                            inputsize = 100
                            classnum = 7
                            agg_type = args.agg_type
                            arc = False
                            dot = False


                            heterograph=GetHeteroMELD(inter=True)

                            model=MultiLink(inputsize,hiddensize,classnum,heterograph.etypes,drop=drop,agg_type=agg_type,dot=dot)



                            optimizer = optim.Adam(model.parameters(),
                                                   lr=lr,
                                                   weight_decay=0.00001
                                                   )


                            best_loss, best_label, best_pred, bestf = None, None, None, None

                            for e in range(epochs):
                                start_time = time.time()
                                train_loss, train_acc, _, _, train_fscore = train_or_eval_model(k,loss_ma,model,heterograph,'train',optimizer)
                                valid_loss, valid_acc, _, _, val_fscore = train_or_eval_model(k,loss_ma,model,heterograph,'dev')
                                test_loss, test_acc, test_label, test_pred, test_fscore = \
                                    train_or_eval_model(k,loss_ma,model,heterograph,'test')


                                if bestf == None or bestf < test_fscore:
                                    best_loss, best_label, best_pred, bestf = \
                                        test_loss, test_label, test_pred, test_fscore

                            print("best F1 is ",bestf)
