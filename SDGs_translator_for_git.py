# coding: utf-8

# Utils =========================================
import random
import numpy as np
import shutil
import os
import pandas as pd
import gc
from datetime import datetime
from os import path
from tqdm import tqdm
import warnings
# For BERT =========================================
import nltk
from nltk.corpus import wordnet
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertModel
from IPython.display import HTML
# For Machine learning =========================================
from sklearn import metrics
from sklearn.manifold import TSNE
# Network analysis =========================================
from networkx.algorithms import community


# Intializing envs ========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_time = '{0:%Y%m%d%H%M}'.format(datetime.now())


# Set Path2Corpus =========================
abs_path = 'YOUR_ROOT_PATH'
#abs_path = r'D:/Dropbox/pj07_sdgs_translator'
path_to_weight = 'YOUR_MODEL_WEIGHT'
#abs_path = r'D:/Dropbox/pj07_sdgs_translator/model_weight_gpu.pth'
path_to_result = os.path.join(abs_path, 'results', current_time)
os.makedirs(path_to_result)
os.makedirs(os.path.join(path_to_result, 'test_attentions'))


# Set variables ==============================
goal_names = np.array(['Goal' + '{:0=2}'.format(goal+1) for goal in range(17)])
class_number = len(goal_names)
goal_contents = np.array(['GOAL 01: No Poverty',
      'GOAL 02: Zero Hunger',
      'GOAL 03: Good Health and Well-being',
      'GOAL 04: Quality Education',
      'GOAL 05: Gender Equality',
      'GOAL 06: Clean Water and Sanitation',
      'GOAL 07: Affordable and Clean Energy',
      'GOAL 08: Decent Work and Economic Growth',
      'GOAL 09: Industry, Innovation and Infrastructure',
      'GOAL 10: Reduced Inequality',
      'GOAL 11: Sustainable Cities and Communities',
      'GOAL 12: Responsible Consumption and Production',
      'GOAL 13: Climate Action',
      'GOAL 14: Life Below Water',
      'GOAL 15: Life on Land',
      'GOAL 16: Peace and Justice Strong Institutions',
      'GOAL 17: Partnerships to achieve the Goal'])


dic_id2cat = dict(zip(list(range(class_number)), goal_contents))
dic_cat2id = dict(zip(goal_names, list(range(class_number))))
dic_cat2id_full = dict(zip(goal_contents, list(range(class_number))))


# Data IO ------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.max_length = max_length
    def __len__(self):
        return len(self.text)
    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            max_length = self.max_length,
            pad_to_max_length = True,
            return_token_type_ids = True,
            truncation = True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


def DataIteratorGenerator(pd, max_length, batch_size):
    pd = pd.reset_index(drop=True)
    dataset = CustomDataset(pd, tokenizer, max_length)
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 0}
    dl = DataLoader(dataset, **params)
    return dl


# Bert ===============================================
class BertForSDGs(nn.Module):
    def __init__(self):
        super(BertForSDGs, self).__init__()
        self.bert = model
        self.cls = nn.Linear(in_features=self.bert.config.hidden_size, out_features=class_number)
    def forward(self, ids, mask, token_type_ids):
        vec, vec_0, attentions = self.bert(ids,
                                        attention_mask = mask,
                                        token_type_ids = token_type_ids,
                                        output_attentions=True,
                                        return_dict=False)
        output = self.cls(vec_0)
        return output, vec_0, attentions


def predictor(net, dl, last_attention_layer):
    texts_list, preds_prob, outputs_list, doc_vecs_list, attentions_list = [], [], [], [], []
    net.eval()
    net.to(device)
    print('now predicting....')
    for batch in tqdm(dl):
        torch.cuda.empty_cache()
        gc.collect()
        ids = batch['ids'].to(device, dtype = torch.long)
        mask = batch['mask'].to(device, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.float)
        for data in range(len(ids)):
            texts_list.append(tokenizer.convert_ids_to_tokens(ids[data]))
        with torch.set_grad_enabled(False):
            outputs, doc_vecs, attentions = net(ids, mask, token_type_ids)
            preds_prob.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            outputs_list.extend(outputs.cpu().detach().numpy().tolist())
            doc_vecs_list.extend(doc_vecs.cpu().detach().numpy().tolist())
            if last_attention_layer==True:
                attentions_list.extend(attentions[-1].cpu().detach().numpy())
            else:
                attentions_batch = []
                for b in range(len(attentions[0])):
                    for l in range(12):
                        if l == 0:
                            attention_sum = attentions[l][b]
                        else:
                            attention_sum += attentions[l][b]
                    attentions_batch.append(attention_sum.cpu().detach().numpy())
                    del attention_sum
                attentions_list.extend(attentions_batch)
    preds_onehot = np.array(preds_prob) >= 0.5
    preds_onehot = preds_onehot.astype(np.int).tolist()
    return preds_prob, preds_onehot, outputs_list, doc_vecs_list, attentions_list


# Attention visualization ============================
def highlight(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def min_max(l):
    l_min = min(l)
    l_max = max(l)
    return [(i - l_min) / (l_max - l_min) for i in l]


def mk_html(net, df, batch_size, max_length, atten_viz_path):
    outputs_iter, doc_vecs_iter, attentions_iter, preds_prob_iter = [], [], [], []
    df_iter = int(np.ceil(df.shape[0]/1000))
    if atten_viz_path != '':
        if os.path.exists(atten_viz_path)==True:
            shutil.rmtree(atten_viz_path)
            os.makedirs(atten_viz_path)
        else:
            os.makedirs(atten_viz_path)
    for d in range(df_iter):
        print('now df_iter=', d, '/', df_iter)
        if d != np.ceil(df.shape[0]/1000):
            sub_df = df.iloc[0+1000*d:1000+1000*d]
        else:
            sub_df = df.iloc[1000*d:-1]
        dl_df = DataIteratorGenerator(sub_dfdf, max_length, batch_size)
        preds_prob, preds_onehot, outputs, doc_vecs, attentions = predictor(net, dl_df, last_attention_layer=False)
        outputs_iter.extend(outputs)
        doc_vecs_iter.extend(doc_vecs)
        preds_prob_iter.extend(preds_prob)
        if atten_viz_path != '':
            print('now html generating...')
            for i, batch in enumerate(tqdm(dl_df)):# i:batch_index
                torch.cuda.empty_cache()
                gc.collect()
                for j in range(len(batch['ids'])):# j:data_index
                    cnt = i*batch_size + j#cnt:df_index
                    label = batch['targets'][j]
                    sentence = batch['ids'][j]
                    pred = preds_onehot[cnt]
                    if sum(list(map(int,label.tolist())))==0:
                        label_str = 'unknown'
                    else:
                        label_str = list(map(int,label.tolist()))
                    html = "Reference: {}<br>Prediction: {}<br>".format(label_str, pred)
                    html += "<br>"
                    for k, goal_content in enumerate(goal_contents):
                        html += highlight(str(goal_content) + "   " + str(float('{:.3f}'.format(preds_prob[cnt][k]))), preds_prob[cnt][k])
                        html += "<br>"
                    html += "<br>"
                    all_attens = torch.zeros(max_length)
                    # all_attention heads -------------------------
                    for l in range(12):
                        all_attens += np.array(attentions[cnt][l][0])
                    all_attens = min_max(all_attens)
                    attentions_iter.append(all_attens)
                    for word, attn in zip(sentence[1:], all_attens[1:]):
                        if tokenizer.convert_ids_to_tokens([word.tolist()])[0] == "[SEP]":
                            html += "<br>"
                            break
                        html += highlight(tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0], attn) + ' '
                    file_num = sum(os.path.isfile(os.path.join(atten_viz_path, name)) for name in os.listdir(atten_viz_path))
                    with open(atten_viz_path + '/attention_viz' + str(file_num) + '.html', 'w', encoding='UTF-8') as res:
                        res.write(html)
    return outputs_iter, doc_vecs_iter, attentions_iter, preds_prob_iter



# tokenizer settings -----------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
criterion = torch.nn.BCEWithLogitsLoss()
nltk.download('wordnet')
nltk.download('omw-1.4')


# Hyper params ===========================
max_length = 2**9
batch_size = 2**5
'''
mux_length must be 512.
batch_size can be changed according to your pc environment
'''


# Model reader ==============================================
if 'net' not in locals():
    net = BertForSDGs()
    net = nn.DataParallel(net, device_ids=[0])
    if torch.cuda.is_available()==True:
        print('on gpu')
        model_path = 'model_gpu.pth'
        net.load_state_dict(torch.load(path_to_weight))
    else:
        model_path = 'model_cpu.pth'
        net.load_state_dict(torch.load(path_to_weight, map_location=torch.device('cpu')))
    net.module.bert.config
    net.module.bert.eval()


# Unknown text prediction =====================================
unknown_texts = [
                'Conduct World-class Research Osaka University will strive to become a vanguard for scholarly research in the world, delving into (1) the truth of matters concerning humans and their varied societies, (2) the truth of all fields related to the environment in which those societies exist, as well as (3) the truth of the interlocking relationships shared by all of the aforementioned.',
                'Promote Advanced Education Osaka University will commit itself to cultivating able and talented persons, persons capable of helping humanity realize its ideals and support society for future generations.',
                'Contribute to Society Osaka University will, through the application of education and research and under its motto of “Live Locally, Grow Globally,” contribute to social stability and welfare, world peace, and the creation of a society in which humans will live in harmony with the natural environment.',
                'Promote Academic Independence and Citizenship In both fields of education and research, Osaka University will, in the traditions of its founding schools Kaitokudo and Tekijuku, continue and advance a free and open-minded citizenry, a citizenry possessing critical thought. Rooted in the essence of academic learning, Osaka University will encourage the spirit of autonomy and independence without flattering any power or authority.',
                'Value Fundamental Research Osaka University will affirm itself as a leader in research for the next generation, focusing on logical and theoretical research, and making global-level research a target.'
                ]


df = pd.DataFrame({'text': unknown_texts, 'label': [[0]*class_number for i in range(len(unknown_texts))]})
output, vecs, attentions, probs = mk_html(net, df, batch_size, max_length, 'test_attentions')


