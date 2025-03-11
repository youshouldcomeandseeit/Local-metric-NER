import json
import numpy as np
import torch

from sklearn.metrics import hamming_loss
def span_offset(offsets_mapping,entities_feature,label_dic):
    '''
    :param offsets_mapping: 每一个句子的offsets_mapping
    :param entities_feature: 句子与之对应的标签列表
    :return: 将对应标签列表转变成经过wordpiece后的索引位置
    '''
    features = entities_feature

    # 后面要写入日志中
    if len(features) != 0:
        for feature in features:
            start_position = feature['start']
            end_position = feature['end']
            '''重复改写！！'''
            if isinstance(feature['type'], int):
                return features
            else:
                feature['type'] = label_dic[feature['type']]

            for index,(start,end) in enumerate(offsets_mapping):
                if start == 0 and end == 0:
                    continue
                if start == start_position:
                    feature['start'] = index

                if end == end_position:
                    feature['end'] = index
    else:
        return []
    return features




def label2id(file_path):
    type_count = {}  # 存储每种type字典
    index = 0
    # type_count['O'] = 0
    with open(file_path, 'r') as f:
        for dic in f:
            data = json.loads(dic)
            entities_features=data['entities']
            for entities in entities_features:
                entity_type = entities['type']
                if entity_type not in type_count:
                    type_count[entity_type] = index
                    index += 1
    return type_count

def co_exist_type(data_list,result,label2list):
    for i in range(len(data_list)):
        for j in range(i+1, len(data_list)):
            if data_list[i]['start'] == data_list[j]['start'] and data_list[i]['end'] == data_list[j]['end'] and data_list[i]['type'] != data_list[j]['type']:
                label_index_x,label_index_y = label2list[data_list[i]['type']],label2list[data_list[j]['type']]
                result[label_index_x,label_index_y] += 1
                result[label_index_y,label_index_x] += 1
    return result
#
def label_counting(data_list,result,label2list):
    for i in range(len(data_list)):
        type1 = label2list[data_list[i]['type']]
        result[type1] +=1
    return result

def Correlation(label2list,data_path):
    Correlation_metric = torch.zeros([len(label2list),len(label2list)],requires_grad=False)
    label_count = torch.zeros(len(label2list))
    with open(data_path, 'r',encoding='utf-8') as fp:
        for line in fp:
            data = json.loads(line)
            entity_mentions = data["entities"]
            Correlation_metric = co_exist_type(entity_mentions, Correlation_metric,label2list)
            label_count = label_counting(entity_mentions,label_count,label2list)
    Cor = Correlation_metric / label_count.unsqueeze(-1)
    Cor = torch.where(torch.isnan(Cor),torch.zeros_like(Cor),Cor)
    return Cor


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()
    def error_analysis(self,y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, s, e in zip(*np.where(y_pred > 0)):
            pred.append((b, l, s, e))

        for b, l, s, e in zip(*np.where(y_true > 0)):
            true.append((b, l, s, e))

        R = set(pred)
        T = set(true)
        conj = R & T
        Pred_wrong = R - conj
        True_wrong = T - conj
        return Pred_wrong,True_wrong

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, s, e in zip(*np.where(y_pred > 0)):
            pred.append((b, l, s, e))

        for b, l, s, e in zip(*np.where(y_true > 0)):
            true.append((b, l, s, e))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)

        print('\n','预测正确：',X,'预测数：',Y,"正确个数",Z)
        return X,Y,Z
            # precision, recall = X / Y, X / Z




def sinusoidal_position_embedding(batch_size, seq_len, output_dim,device):
    position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

    indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    indices = torch.pow(10000, -2 * indices / output_dim)
    embeddings = position_ids * indices
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings

def rope(qw,kw):
    device = qw.device
    b,seq,dim = qw.shape[0],qw.shape[1],qw.shape[-1]
    pos_emb = sinusoidal_position_embedding(b, seq, dim,device)
    # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
    cos_pos = pos_emb[... , None, 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[... , None, ::2].repeat_interleave(2, dim=-1)
    qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
    qw2 = qw2.reshape(qw.shape)
    qw = qw * cos_pos + qw2 * sin_pos
    kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
    kw2 = kw2.reshape(kw.shape)
    kw = kw * cos_pos + kw2 * sin_pos

    return qw,kw

