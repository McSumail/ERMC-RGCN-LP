import numpy as np
import torch
import dgl
import pickle
speaker_tag={'Alice': 0, 'Charlie': 1, 'Carol': 2, 'Ben': 3, 'Emily': 4, 'Monica': 5, 'Mike': 6, 'Richard': 7,\
             'Julie': 8, 'Barry': 9, 'Dina': 10, 'David': 11, 'Leslie': 12, 'Kathy': 13, 'Ross': 14, 'Rachel': 15,\
             'Phoebe': 16, 'Chandler': 17, 'Joey': 18, 'Mark': 19, 'Ursula': 20, 'Mindy': 21, 'Susan': 22,'other':23}
speaker_relation={'MikePhoebe': 5, 'PhoebeMike': 5, 'CarolSusan': 5, 'SusanCarol': 5, 'CarolBen': 3, \
                  'BenCarol': 0, 'MonicaChandler': 5, 'ChandlerMonica': 5, 'JoeyUrsula': 0, 'RachelRoss': 0, \
                  'RossEmily': 5, 'EmilyRoss': 5, 'BarryMindy': 5, 'MindyBarry': 5, 'EmilyRachel': 2, \
                  'AliceLeslie': 0, 'AliceChandler': 0, 'LeslieAlice': 3, 'ChandlerAlice': 3, 'MonicaRichard': 5,\
                  'RichardMonica': 5, 'RachelMark': 0, 'MarkRachel': 0, 'DinaJoey': 6, 'JoeyDina': 6, 'JoeyCharlie': 0,\
                  'JulieRoss': 5, 'RossJulie': 5, 'RossRachel': 1, 'DavidPhoebe': 5, 'PhoebeDavid': 5, 'CharlieJoey': 5,\
                  'MonicaRachel': 7, 'RachelMonica': 7, 'JoeyRachel': 0, 'KathyJoey': 5, 'JoeyKathy': 5}
relation_tag={'parents/child': 0, 'girl/boyfriend': 1, 'negative_impression': 2, 'friends': 3, 'roommate': 4, \
              'positive_impression': 5, 'spouse': 6, 'siblings': 7, 'none': 8}

def GetHeteroMELD(inter=False):
    text, speaker, emotion, sentiment, trainid, devid, testid = pickle.load(open("../examples/MELD/BERT_speaker23.pkl", 'rb'))
    num_utt,num_speaker=13708,24
    train_mask,dev_mask,test_mask=9989,11098,13708
    allid=trainid|devid|testid

    #speaker
    speak_src,speak_dst=[],[]
    num=0
    for id in allid:
        len_dia=len(text[id])
        for i in range(num,num+len_dia):
            speak_src.append(speaker[id][i-num])
            speak_dst.append(i)
        num+=len_dia
    speaker_src, speaker_dst = np.array(speak_src), np.array(speak_dst)

    #conversational context
    follow_src,follow_dst=[],[]
    num=0
    for id in allid:
        len_dia=len(text[id])
        for i in range(num,num+len_dia-1):
            follow_dst.append(i)
            follow_src.append(i+1)
        num+=len_dia
    follow_src, follow_dst = np.array(follow_src), np.array(follow_dst)


    #interaction
    if inter==True:
        inter_src, inter_dst = [], []
        tuple_set = set()
        num = 0
        for id in allid:
            len_dia=len(text[id])
            my_set=[]
            for i in range(num,num+len_dia):
                if speaker[id][i-num] not in my_set:
                    my_set.append(speaker[id][i-num])
            for i in range(len(my_set)):
                for j in range(len(my_set)):
                    if i==j:continue
                    tuple_set.add((my_set[i],my_set[j]))
            num+=len_dia
        for item in tuple_set:
            inter_src.append(item[0])
            inter_dst.append(item[1])
        inter_src, inter_dst = np.array(inter_src), np.array(inter_dst)

    if  inter:
        hetero_graph=dgl.heterograph({
            ('utt','follow','utt'):(follow_src,follow_dst),
            ('utt','followed-by','utt'):(follow_dst,follow_src),
            ('speaker','speak','utt'):(speaker_src,speaker_dst),
            ('utt','spoken-by','speaker'):(speaker_dst,speaker_src),
            ('speaker','interact-with','speaker'):(inter_src,inter_dst)
        })
    else:
        hetero_graph=dgl.heterograph({
            ('utt','follow','utt'):(follow_src,follow_dst),
            ('utt','followed-by','utt'):(follow_dst,follow_src),
            ('speaker','speak','utt'):(speaker_src,speaker_dst),
            ('utt','spoken-by','speaker'):(speaker_dst,speaker_src),
        })



    features,emotionlabels,sentimentlabels=[],[],[]
    for id in allid:
        for i in range(len(text[id])):
            features.append(text[id][i])
            emotionlabels.append(emotion[id][i])
            sentimentlabels.append(sentiment[id][i])


    tr_mask=torch.zeros(num_utt,dtype=torch.bool)
    de_mask=torch.zeros(num_utt,dtype=torch.bool)
    te_mask=torch.zeros(num_utt,dtype=torch.bool)
    tr_mask[:train_mask]=1
    de_mask[train_mask:dev_mask]=1
    te_mask[dev_mask:test_mask]=1
    hetero_graph.nodes['utt'].data['feature']=torch.FloatTensor(features)
    hetero_graph.nodes['speaker'].data['feature']=torch.randn(num_speaker,len(features[0]))
    hetero_graph.nodes['utt'].data['emotionlabel']=torch.LongTensor(emotionlabels)
    hetero_graph.nodes['utt'].data['sentimentlabel']=torch.LongTensor(sentimentlabels)
    hetero_graph.nodes['utt'].data['trainmask']=tr_mask
    hetero_graph.nodes['utt'].data['devmask']=de_mask
    hetero_graph.nodes['utt'].data['testmask']=te_mask

    return hetero_graph

def GetHeteroEmory(inter=False):
    text, speaker, emotion, trainid, testid = pickle.load(open("../examples/EmoryNLP/BERT.pkl", 'rb'))
    num_utt, num_speaker=9489,7
    train_mask, dev_mask, test_mask=7551,8505,9489
    allid=trainid|testid

    #speaker
    speak_src,speak_dst=[],[]
    num=0
    for id in allid:
        len_dia=len(text[id])
        for i in range(num,num+len_dia):
            speak_src.append(speaker[id][i-num])
            speak_dst.append(i)
        num+=len_dia
    speaker_src, speaker_dst = np.array(speak_src), np.array(speak_dst)

    #conversational context
    follow_src,follow_dst=[],[]
    num=0
    for id in allid:
        len_dia=len(text[id])
        for i in range(num,num+len_dia-1):
            follow_dst.append(i)
            follow_src.append(i+1)
        num+=len_dia
    follow_src, follow_dst = np.array(follow_src), np.array(follow_dst)


    #interaction
    if inter==True:
        inter_src, inter_dst = [], []
        tuple_set = set()
        num = 0
        for id in allid:
            len_dia=len(text[id])
            my_set=[]
            for i in range(num,num+len_dia):
                if speaker[id][i-num] not in my_set:
                    my_set.append(speaker[id][i-num])
            for i in range(len(my_set)):
                for j in range(len(my_set)):
                    if i==j:continue
                    tuple_set.add((my_set[i],my_set[j]))
            num+=len_dia
        for item in tuple_set:
            inter_src.append(item[0])
            inter_dst.append(item[1])
        inter_src, inter_dst = np.array(inter_src), np.array(inter_dst)

    if  inter:
        hetero_graph=dgl.heterograph({
            ('utt','follow','utt'):(follow_src,follow_dst),
            ('utt','followed-by','utt'):(follow_dst,follow_src),
            ('speaker','speak','utt'):(speaker_src,speaker_dst),
            ('utt','spoken-by','speaker'):(speaker_dst,speaker_src),
            ('speaker','interact-with','speaker'):(inter_src,inter_dst)
        })
    else:
        hetero_graph=dgl.heterograph({
            ('utt','follow','utt'):(follow_src,follow_dst),
            ('utt','followed-by','utt'):(follow_dst,follow_src),
            ('speaker','speak','utt'):(speaker_src,speaker_dst),
            ('utt','spoken-by','speaker'):(speaker_dst,speaker_src),
        })



    features,emotionlabels=[],[]
    for id in allid:
        for i in range(len(text[id])):
            features.append(text[id][i])
            emotionlabels.append(emotion[id][i])



    tr_mask=torch.zeros(num_utt,dtype=torch.bool)
    de_mask=torch.zeros(num_utt,dtype=torch.bool)
    te_mask=torch.zeros(num_utt,dtype=torch.bool)
    tr_mask[:train_mask]=1
    de_mask[train_mask:dev_mask]=1
    te_mask[dev_mask:test_mask]=1
    hetero_graph.nodes['utt'].data['feature']=torch.FloatTensor(features)
    hetero_graph.nodes['speaker'].data['feature']=torch.randn(num_speaker,len(features[0]))
    hetero_graph.nodes['utt'].data['emotionlabel']=torch.LongTensor(emotionlabels)
    hetero_graph.nodes['utt'].data['trainmask']=tr_mask
    hetero_graph.nodes['utt'].data['devmask']=de_mask
    hetero_graph.nodes['utt'].data['testmask']=te_mask

    return hetero_graph




