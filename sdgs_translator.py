# coding: utf-8

import sys
import shutil
import random
import math
import numpy as np
import os
import platform
import re
import string
import pandas as pd
import pickle
import optuna
import gc
import time
import multiprocessing
from langdetect import detect
from datetime import datetime
from os import listdir, path
from tqdm import tqdm
from googletrans import Translator
# For Bert =========================================
import MeCab
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw')
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn
import torch.optim as optim
import transformers
from transformers import BertTokenizer, BertModel, BertConfig, BertForPreTraining
from transformers.modeling_bert import BertModel
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from IPython.display import display, HTML
import webbrowser
# For Machine learning =========================================
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Network analysis
from networkx.algorithms import community
import networkx as nx
import matplotlib.pyplot as plt


sys.path.append('../../')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Set Path2Corpus =========================
np.set_printoptions(precision=3)

if os.name == 'nt':
    abs_path = 'D:/Dropbox/pj07_sdgs_translator'
elif os.name == 'posix':
    abs_path = '/Users/mt/Dropbox/GE/pj07_sdgs_translator'
    abs_path = '/home/ge/Dropbox/pj07_sdgs_translator'


path_to_corpus = os.path.join(abs_path, 'corpus_zoo')
current_time = '{0:%Y%m%d%H%M}'.format(datetime.now())
path_to_result = os.path.join(abs_path, 'results', current_time)

if os.getcwd()=='results':
    os.makedirs(path_to_result)
    os.chdir(path_to_result)
    os.makedirs('train_attentions')
    os.makedirs('val_attentions')
    os.makedirs('test_attentions')


# Set variables ==============================
goal_names = np.array(['Goal' + '{:0=2}'.format(goal+1) for goal in range(17)])
class_number = len(goal_names)
goal_contents = np.array([
      'GOAL 01: No Poverty',
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


sdgs_colors = ['#D60036','#D5A428','#5EA342','#B90029','#E0291D','#55C0E9','#F4C200','#970043','#E46018','#D10081','#ED9800','#B88A1B','#4E8145','#4497D6','#6DBD40','#306AA0','#264B6C']
dic_id2cat = dict(zip(list(range(class_number)), goal_contents))
dic_cat2id = dict(zip(goal_names, list(range(class_number))))
dic_cat2id_full = dict(zip(goal_contents, list(range(class_number))))



def save_params(param_names, params_memory, name):
    col_names = np.array(param_names).reshape(1,-1)
    params = pd.DataFrame(params_memory, columns = col_names[0])
    filename = 'bert_best_params.csv'
    params.to_csv(filename, index=False, header=True)


def outer_df_saver(train_df, test_df, cv):
    train_df.to_csv("./train.csv", index=False, header=None)
    test_df.to_csv("./test.csv", index=False, header=None)


def inner_df_saver(train_df, test_df, cv):
    train_df.to_csv("./val_train.csv", index=False, header=None)
    test_df.to_csv("./val.csv", index=False, header=None)


# Machine Learning =========================
def train_test_splitter(df, cvs, cv):
    df = df.reset_index(drop=True)
    test_idx = []
    for n in list(np.array_split(list(range(0,10)), cvs))[cv]:
        test_idx.extend(list(df.index[n::10]))
#    test_idx = list(df.index[cv::cvs])
    train_idx = set(df.index)-set(test_idx)
    test_df = df.loc[test_idx]
    train_df =df.loc[train_idx]
    return train_df, test_df


# Pre-processing ============================
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



def DataIteratorGenerator(train_pd, test_pd, max_length, batch_size):
    train_pd = train_pd.reset_index(drop=True)
    test_pd = test_pd.reset_index(drop=True)
    training_set = CustomDataset(train_pd, tokenizer, max_length)
    testing_set = CustomDataset(test_pd, tokenizer, max_length)
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }
    test_params = {'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 0
                    }
    dl_train = DataLoader(training_set, **train_params)
    dl_test = DataLoader(testing_set, **test_params)
    return dl_train, dl_test


def wordnet_augmenter(sentence, mutation_ratio, erase_ratio):
    # Mutation ==============================================
    m = MeCab.Tagger('-Owakati')
    sentence = m.parse(sentence).split(' ')
    wordnet_sentence = [w for w in sentence]
    num = len(sentence)
    hit = int(num * mutation_ratio) + 1
    replacer = random.sample(list(range(num)), hit)
    for w in replacer:
        word = sentence[w]
        if bool(re.search(r'[a-zA-Z0-9]', word)):
            synsets = wordnet.synsets(word, lang='eng')
        else:
            synsets = wordnet.synsets(word, lang='jpn')
        # Replace words ==============================================a
        if len(synsets) != 0 and synsets != '':
            synonims = synsets[0].lemma_names("jpn")
            if len(synonims) != 0 and synonims != '':
                target = random.choice(list(range(len(synonims))))
                obj_wordnet = synonims[target]
            else:
                obj_wordnet = word
        else:
            obj_wordnet = word
        wordnet_sentence[w] = obj_wordnet
   # Erase words ==============================================a
    if num >= 10:
        for i in range(int(num*erase_ratio)+1):
            l = len(wordnet_sentence)
            out = random.choice(list(range(0, l)))
            del wordnet_sentence[out]
    return ''.join(wordnet_sentence)


def train_df_augmentation(train_df, augmentation_times):
    augmented_train_df = train_df
    for time in range(augmentation_times):
        for idx, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
            augmented_text = wordnet_augmenter(row['text'], mutation_ratio=0.1, erase_ratio=0.1)
            augmented_train_df = augmented_train_df.append({'text':augmented_text,'label':row['label']}, ignore_index=True)
    return augmented_train_df


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
                                        output_attentions=True)
        output = self.cls(vec_0)
        return output, vec_0, attentions


def net_initializer(net):
    for param in net.parameters():
        param.requires_grad = False
    for param in net.bert.encoder.layer[-1].parameters():
        param.requires_grad = True
    for param in net.cls.parameters():
        param.requires_grad = True
    return net


def bert_train(net, dataloaders_dict, criterion, optimizer, num_epochs, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    torch.backends.cudnn.benchmark = True
    batch_size = batch_size
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0
            iteration = 1
            for batch in (dataloaders_dict[phase]):
                ids = batch['ids'].to(device, dtype = torch.long)
                mask = batch['mask'].to(device, dtype = torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
                targets = batch['targets'].to(device, dtype = torch.float)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _, _ = net(ids, mask, token_type_ids)
                    optimizer.zero_grad()
                    loss = criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item()*batch_size
            epoch_loss = epoch_loss/(len(dataloaders_dict[phase])*batch_size)
            if phase == 'train':
                epoch_train_loss = epoch_loss
            else:
                epoch_val_loss = epoch_loss
        print('Epoch {}/{} | train_loss: {:.3f} val_loss: {:.3f}'.format("{0:02d}".format(epoch+1), num_epochs, epoch_train_loss, epoch_val_loss))
    return net


def bert_best_train(net, dataloaders, criterion, optimizer, num_epochs, batch_size, early_stopping_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    torch.backends.cudnn.benchmark = True
    batch_size = batch_size
    early_stopping_counter = 0
    loss_memory = 0
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        epoch_corrects = 0
        for batch in (dataloaders):
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.float)
            with torch.set_grad_enabled(True):
                outputs, _, _ = net(ids, mask, token_type_ids)
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_size
        epoch_loss = epoch_loss / (len(dataloaders)*batch_size)
        loss_memory = epoch_loss
        print('Epoch {}/{} | train loss: {:.3f}'.format("{0:02d}".format(epoch+1), num_epochs, epoch_loss))
        if loss_memory < epoch_loss:
            early_stopping_counter += 1
        if early_stopping_counter >= int(batch_size*early_stopping_rate):
            print('Epoch {}/{} | train loss: {:.3f}'.format("{0:02d}".format(epoch+1), num_epochs, epoch_loss))
            print('Early stopping executed...')
            break
    return net


def opt_bert(trial):
    max_length = trial.suggest_int('max_length', 9, 9)
    batch_size = trial.suggest_int('batch_size', 3, 6)
    encoder_lr = trial.suggest_loguniform('encoder_lr', 1e-5, 1e-2)
    cls_lr = trial.suggest_loguniform('cls_lr', 1e-5, 1e-2)
    test_loss_list = []
    for inner_cv in range(0, inner_cvs):
        val_train_df, val_df = train_test_splitter(train_df, inner_cvs, inner_cv)
        if augmentation_times != 0:
            val_train_df = train_df_augmentation(val_train_df, augmentation_times)
        print('outer_cv' + str(outer_cv) + ', inner_cv' + str(inner_cv) + '_datasize=' + str(len(val_train_df)) + '_processing...')
        global counter, previous_time
        counter += 1
        previous_time = datetime.now()
        dl_val_train, dl_val = DataIteratorGenerator(val_train_df, val_df, 2**max_length, 2**batch_size)
        inner_dataloaders_dict = {"train": dl_val_train, "val": dl_val}
        inner_net = BertForSDGs()
        inner_net.train()
        inner_net = net_initializer(inner_net)
        optimizer = optim.Adam([
                                {'params': inner_net.bert.encoder.layer[-1].parameters(), 'lr': encoder_lr},
                                {'params': inner_net.cls.parameters(), 'lr': cls_lr}
                                ])
        inner_net_trained = bert_train(inner_net, inner_dataloaders_dict, criterion, optimizer, num_epochs=epochs, batch_size=2**batch_size)
        loss, accuracy, f1_score_micro, f1_score_macro, _, _, _, _ = test_eval(inner_net_trained, dl_val)
        test_loss_list.append(loss)
        print('loss：{:.4f}, acc：{:.4f}, f1_score_micro：{:.4f}, f1_score_macro：{:.4f}'.format(loss, accuracy, f1_score_micro, f1_score_macro))
        print((max_times - counter), '/', max_times, ' finish on ', (max_times - counter)*(datetime.now()-previous_time)+datetime.now())
    return np.mean(test_loss_list)


def test_eval(net, dl):
    targets_list, texts_list, preds_prob, losses = [], [], [], []
    net.eval()
    net.to(device)
    for batch in dl:
        ids = batch['ids'].to(device, dtype = torch.long)
        mask = batch['mask'].to(device, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.float)
        for data in range(len(ids)):
            texts_list.append(tokenizer.convert_ids_to_tokens(ids[data]))
        with torch.set_grad_enabled(False):
            outputs, _, _ = net(ids, mask, token_type_ids)
            loss = criterion(outputs, targets)
            targets_list.extend(targets.cpu().detach().numpy().tolist())
            preds_prob.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            losses.append(float(loss))
    preds_onehot = np.array(preds_prob) >= 0.5
    preds_onehot = preds_onehot.astype(np.int).tolist()
    accuracy = metrics.accuracy_score(targets_list, preds_onehot)
    f1_score_micro = metrics.f1_score(targets_list, preds_onehot, average='micro')
    f1_score_macro = metrics.f1_score(targets_list, preds_onehot, average='macro')
    return np.mean(losses), accuracy, f1_score_micro, f1_score_macro, targets_list, texts_list, preds_prob, preds_onehot


def unknown_predictor(net, dl, last_attention_layer):
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
    attn = attn*0.8
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def min_max(l):
    l_min = min(l)
    l_max = max(l)
    return [(i - l_min) / (l_max - l_min) for i in l]


def mk_html(net, df, batch_size, max_length, atten_viz_path):
    outputs_iter, doc_vecs_iter, attentions_iter, preds_prob_iter = [], [], [], []
    df_iter = int(np.ceil(df.shape[0]/1000))
    if os.path.exists(atten_viz_path)==True:
        shutil.rmtree(atten_viz_path)
        os.makedirs(atten_viz_path)
    else:
        os.makedirs(atten_viz_path)
    for d in range(df_iter):
        print('now df_iter=', d)
        if d != np.ceil(df.shape[0]/1000):
            sub_df = df.iloc[0+1000*d:1000+1000*d]
        else:
            sub_df = df.iloc[1000*d:-1]
        labels = list(sub_df.label)
        sentences = list(sub_df.text)
        _, dl_df = DataIteratorGenerator(sub_df, sub_df, max_length, batch_size)
        preds_prob, preds_onehot, outputs, doc_vecs, attentions = unknown_predictor(net, dl_df, last_attention_layer=False)
        outputs_iter.extend(outputs)
        doc_vecs_iter.extend(doc_vecs)
        preds_prob_iter.extend(preds_prob)
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
        #ENG                html += highlight(tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0], attn) + ' '
                    html += highlight(tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0], attn)
                file_num = sum(os.path.isfile(os.path.join(atten_viz_path, name)) for name in os.listdir(atten_viz_path))
                with open(atten_viz_path + '/attention_viz' + str(file_num) + '.html', 'w', encoding='UTF-8') as res:
                    res.write(html)
    return outputs_iter, doc_vecs_iter, attentions_iter, preds_prob_iter



# --------------------------------------------------
def get_similarity_ranking(input_vec, target_vec_list, rank=10):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cossim_list = []
    for target_vec in target_vec_list:
        output = cos(input_vec, target_vec)
        cossim_list.append(output)



# tokenizer settings -----------------------------------------
'''
In our article case, 
tokenizer_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
'''
tokenizer = BertJapaneseTokenizer.from_pretrained(tokenizer_name)
model = BertModel.from_pretrained(model_name)
criterion = torch.nn.BCEWithLogitsLoss()
nltk.download('wordnet')


# ==============================================================
# main =========================================================
# Data loader ---------------------------------------------------------
'''
texts = list of texts ['text1', 'text2', 'text3', ....]
text* = 'foobarfoobarfoobarfoobarfoobar.'

labels = list of multihot vectors [vector1, vector2, vector3, ....] 
vector* = [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1]
'''
# Generate Data frame ---------------------------------------------------------
# Extend Data frame ---------------------------------------------------------
extended_texts = []
extended_labels = []
for label, text in zip(labels, texts):
    div = int(np.ceil(len(text)/512))
    for i in range(div):
        extended_texts.append(text[0+i*512:(i+1)*512])
        extended_labels.append(label)

# Generate Data frame ---------------------------------------------------------
df = pd.DataFrame({'text' : extended_texts, 'label' : extended_labels})
df = df.sample(frac=1)


# Hyper params ===========================
outer_cvs =  10 # cv num
inner_cvs =  5 # cv num
n_trials = 10
epochs = 2**4
best_epochs = 2**5
augmentation_times = 1
max_length = 2**9
batch_size = 2**3


# end time expector -----------------
max_times = outer_cvs*n_trials*inner_cvs
counter = 0
previous_time = datetime.now()


# Memory initializeing ====================
train_loss_list, train_accuracy_list, train_f1_score_micro_list, train_f1_score_macro_list, test_loss_list, test_accuracy_list, test_f1_score_micro_list, test_f1_score_macro_list = [], [], [], [], [], [], [], []
bert_param_names = np.array(['max_length', 'batch_size', 'encoder_lr', 'cls_lr']).reshape(1,-1)
bert_val_params_memory = np.zeros((outer_cvs, bert_param_names.shape[1]))
bert_test_params_memory = np.zeros((outer_cvs, bert_param_names.shape[1]))
goals_list, texts_list, preds_prob_list, preds_onehot_list, plain_texts_list = [], [], [], [], []
# Training ===============================
for outer_cv in range(outer_cvs):
    train_df, test_df = train_test_splitter(df, outer_cvs, outer_cv)
    plain_texts_list.extend(test_df.text.to_list())
    study = optuna.create_study()
    study.optimize(opt_bert, n_trials = n_trials)
    for param in range(len(bert_param_names[0])):
        bert_test_params_memory[outer_cv, param] = study.best_params[bert_param_names[0,param]]
    if augmentation_times != 0:
        train_df = train_df_augmentation(train_df, augmentation_times)
    dl_train, dl_test = DataIteratorGenerator(train_df, test_df,
                                            2**(study.best_params['max_length']),
                                            2**(study.best_params['batch_size']))
    outer_dataloaders_dict = {"train": dl_train, "val": dl_test}
    outer_net = BertForSDGs()
    outer_net.train()
    outer_net = net_initializer(outer_net)
    optimizer = optim.Adam([
                {'params': outer_net.bert.encoder.layer[-1].parameters(), 'lr': study.best_params['encoder_lr']},
                {'params': outer_net.cls.parameters(), 'lr': study.best_params['cls_lr']}
    ])
    outer_net_trained = bert_train(outer_net, outer_dataloaders_dict,
                        criterion, optimizer, num_epochs=epochs, batch_size=2**study.best_params['batch_size'])
    train_loss, train_accuracy, train_f1_score_micro, train_f1_score_macro, _, _, _, _ = test_eval(outer_net_trained, dl_train)
    test_loss, test_accuracy, test_f1_score_micro, test_f1_score_macro, goals, texts, preds_prob, preds_onehot = test_eval(outer_net_trained, dl_test)
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)
    train_f1_score_micro_list.append(train_f1_score_micro)
    train_f1_score_macro_list.append(train_f1_score_macro)
    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy)
    test_f1_score_micro_list.append(test_f1_score_micro)
    test_f1_score_macro_list.append(test_f1_score_macro)
    goals_list.extend(goals)
    texts_list.extend(texts)
    preds_prob_list.extend(preds_prob)
    preds_onehot_list.extend(preds_onehot)
    _, _, _, _ = mk_html(outer_net, test_df, batch_size, max_length, 'val_attentions')
    with open('cv_results.txt','a', encoding='utf-8') as res:
        res.write(','.join([str(train_loss), str(test_loss), str(train_accuracy), str(test_accuracy), str(train_f1_score_micro), str(test_f1_score_micro), str(train_f1_score_macro), str(test_f1_score_macro)]) + '\n')


save_params(bert_param_names, bert_test_params_memory, 'bert')
results = np.hstack([np.array(plain_texts_list).reshape(-1,1), np.array(goals_list).reshape(-1,class_number), np.array(preds_onehot_list).reshape(-1,class_number), np.array(preds_prob_list).reshape(-1,class_number), np.array(texts_list).reshape(-1,2**9)])
title = ['text']
title_obs = ['obs' + str(i) for i in range(class_number)]
title_onehot = ['onehot' + str(i) for i in range(class_number)]
title_prob = ['prob' + str(i) for i in range(class_number)]
title_text = ['text' + str(i) for i in range(2**9)]
title = title + title_obs + title_onehot + title_prob + title_text
results_df = pd.DataFrame(results, columns=title)
results_df.to_csv('results.csv')
results_df = pd.read_csv('results.csv', header=0)
y_pred = np.array(results_df.iloc[:,19:36].astype(float).astype(int))
y_true = np.array(results_df.iloc[:,2:19].astype(float).astype(int))
metrics.confusion_matrix(y_true, y_pred)
print(metrics.classification_report(y_true,y_pred))
classification_report = metrics.classification_report(y_true,y_pred)
with open('classification_report.txt','w', encoding='utf-8') as res:
    res.write(classification_report)


# Best model training ======================================
bert_test_params_memory = pd.read_csv('bert_best_params.csv', header=0)
bert_param_names = np.array(['max_length', 'batch_size', 'encoder_lr', 'cls_lr']).reshape(1,-1)
bert_mean_best_params = {name: param for name, param in zip(bert_param_names[0], np.mean(bert_test_params_memory, axis=0))}
if augmentation_times != 0:
    best_train_df = train_df_augmentation(df, augmentation_times)
else:
    best_train_df = df

dl_best_train, _ = DataIteratorGenerator(df, df,
                                        2**(int(bert_mean_best_params['max_length'])),
                                        2**(int(bert_mean_best_params['batch_size'])))
outer_dataloaders_dict = {"train": dl_best_train, "val": dl_test}
best_net = BertForSDGs()
best_net.train()
best_net = net_initializer(best_net)
optimizer = optim.Adam([
                {'params': best_net.bert.encoder.layer[-1].parameters(), 'lr': bert_mean_best_params['encoder_lr']},
                {'params': best_net.cls.parameters(), 'lr': bert_mean_best_params['cls_lr']}
            ], betas=(0.9, 0.999))
best_net = bert_best_train(best_net,
              dl_best_train,
              criterion,
              optimizer,
              num_epochs=best_epochs,
              batch_size=2**math.ceil(bert_mean_best_params['batch_size']),
              early_stopping_rate=0.1)


# Best model reader and loader ==============================================
if os.getcwd()=='results':
    model_path = 'best_model_gpu.pth'
    torch.save(best_net.state_dict(), model_path)
    model_path = 'best_model_cpu.pth'
    torch.save(best_net.to('cpu').state_dict(), model_path)


best_net = BertForSDGs()
if torch.cuda.is_available()==True:
    model_path = 'best_model_gpu.pth'
    best_net.load_state_dict(torch.load(model_path))
else:
    model_path = 'best_model_cpu.pth'
    best_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


# Train attention visualization --------------------------------
o,d1,a,preds_prob = mk_html(best_net, df, batch_size, max_length, 'train_attentions')


# Unknown text predictor =====================================
unknown_text1 = 'spamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspamspam'
unknown_text2 = 'hamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamhamham'
unknown_text3 = 'eggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggseggs'


unknown_texts = [unknown_text1, unknown_text2, unknown_text3]
unknown_df = pd.DataFrame({'text': unknown_texts, 'label': [[0]*class_number for i in range(len(unknown_texts))]})
o,d2,a, preds_prob = mk_html(best_net, unknown_df, batch_size, max_length, 'unknown_attentions')


# Nexus visualizer ---------------------------------------------------
import itertools
from sklearn.metrics import jaccard_score
preds_onehot=np.array(preds_prob) >= 0.5
data = pd.DataFrame(np.array(preds_onehot))
l = np.arange(0,17)
C = list(itertools.combinations(l,2))
jaccards = []
for c in C:
    jaccards.append(jaccard_score(data[c[0]], data[c[1]]))
network = []
for c in C:
    jaccard = (jaccard_score(data[c[0]], data[c[1]])-np.min(jaccards))/(np.max(jaccards)-np.min(jaccards))
    network.append((c[0], c[1], jaccard))


# Build network ---------------------------------------
G = nx.Graph()
G.add_nodes_from(l, size=10)
for i, j, w in network:
    G.add_edge(i, j, weight=w)

communities_generator = community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
sorted(map(sorted, next_level_communities))


# nx.write_gml(G, "pagerank.gml")
plt.figure(figsize=(10,20), dpi=100)
pos = nx.spring_layout(G)
pr = nx.pagerank(G)
node_size = [(120*v)**3 for v in pr.values()]
nx.draw_networkx_nodes(G, pos, node_color=sdgs_colors, alpha=0.4, node_size=node_size)
nx.draw_networkx_labels(G, pos, labels=dict(enumerate(goal_names,0)), font_size=10)
edge_width = [(d["weight"]*10)**1.5 for (u,v,d) in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="grey", width=edge_width)
plt.axis('off')
plt.savefig('indicator_nexus.png')



# Match making part --------------------------------------------
################################
# seeds
'''
seeds_texts, needs_texts = list of texts ['text1', 'text2', 'text3', ....]
text* = 'foobarfoobarfoobarfoobarfoobar.'
seeds_labels, needs_labels = list of multihot vectors [vector1, vector2, vector3, ....] 
vector* = [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1]
'''

# シーズをデータフレーム化、ベクトル取得
seeds_df = pd.DataFrame({'text': seeds_texts, 'label': seeds_labels})
_, seed_vecs, _, seed_probs = mk_html(best_net, seeds_df, batch_size, max_length, 'seeds_attentions')
needs_df = pd.DataFrame({'text': needs_texts, 'label': needs_labels})
_, need_vecs, _, need_probs  = mk_html(best_net, needs_df, batch_size, max_length, 'needs_attentions')


# Cosine similarity ----------------------
need_vec = need_vecs[1]
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
cossims = []
for_freq = []
for i, seed_vec in enumerate(seed_vecs):
    output = cos(torch.tensor(need_vec), torch.tensor(seed_vec))
    res = [i, output.item()]
    cossims.append(res)
    for_freq.append(output.item())

cossim_df = pd.DataFrame(cossims)
df_sim_values = pd.concat([cossim_df, seeds_df['text']], axis=1)
df_sim_values = df_sim_values.rename({0: "id", 1: "cossim"}, axis='columns')
df_sim_top3 = df_sim_values.sort_values(by=['cossim'], ascending=False)[0:3]
df_sim_worst3 = df_sim_values.sort_values(by=['cossim'], ascending=True)[0:3]
df_sim_values.to_csv('df_sim_values.csv')


# Clustering by t-SNE ==============================================
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import seaborn as sns


seed_vecs.extend(need_vecs)
all_vecs = seed_vecs.copy()
seed_vecs = seed_vecs[0:len(seeds_df)]
all_vecs_reduced = TSNE(n_components=2, random_state=42, perplexity=5).fit_transform(all_vecs)
seed_vecs_reduced = all_vecs_reduced[0:-(len(need_vecs))]
need_vecs_reduced = all_vecs_reduced[-(len(need_vecs)):]
seed_goals = [np.argmax(seed_prob) for seed_prob in seed_probs]
need_goals = [np.argmax(need_prob) for need_prob in need_probs]

df_seeds = pd.DataFrame(np.c_[np.array(seed_goals).reshape(-1,1), seed_vecs_reduced])
df_seeds.columns = ['goal','Primary Dimension','Secondary Dimension']
df_needs = pd.DataFrame(np.c_[np.array(need_goals).reshape(-1,1), need_vecs_reduced])
df_needs.columns = ['goal','Primary Dimension','Secondary Dimension']
plt.figure(figsize=(30, 30), dpi=100)
g = sns.scatterplot(data=df_seeds,
                    x='Primary Dimension',
                    y='Secondary Dimension',
                    hue='goal',
                    linewidth=0,
                    alpha = 0.7,
                    s=50,
                    palette=sdgs_colors[0:-1])
g = sns.scatterplot(data=df_needs,
                    x='Primary Dimension',
                    y='Secondary Dimension',
                    hue='goal',
                    linewidth=0,
                    alpha = 0.7,
                    s=1000)

for i in df_seeds.index:
    g.text(df_seeds.iloc[i,1], df_seeds.iloc[i,2], str(i))

for i in df_needs.index:
    g.text(df_needs.iloc[i,1]+.02, df_needs.iloc[i,2], 'Needs' + str(i))

handles, _ = g.get_legend_handles_labels()
plt.legend(handles, goal_contents, bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0, ncol=1)
plt.setp(g.get_legend().get_texts(), fontsize=5)
plt.savefig('tsne_goal.png',bbox_inches='tight')
