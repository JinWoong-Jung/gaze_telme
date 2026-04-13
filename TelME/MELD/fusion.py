import glob
import json
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc
import os

from preprocessing import *
from utils import *
from dataset import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
use_amp = (device.type == "cuda")

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CELoss(pred_outs, labels):
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def model_train(training_epochs, model_t, audio_s, video_s, fusion, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path, results_path=None):
    best_dev_fscore, best_test_fscore = 0, 0
    best_metrics = {}
    best_epoch = 0

    for epoch in tqdm(range(training_epochs)):
        fusion.train() 
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                """Prediction"""
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.to(device), attention_masks.to(device), audio_inputs.to(device), video_inputs.to(device), batch_labels.to(device)

                text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
                audio_hidden, audio_logits = audio_s(audio_inputs)
                video_hidden, video_logits = video_s(video_inputs)

                pred_logits = fusion(text_hidden, audio_hidden, video_hidden)

                loss_val = CELoss(pred_logits, batch_labels)

            if use_amp:
                scaler.scale(loss_val).backward()
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            else:
                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

        fusion.eval()   
        dev_pred_list, dev_label_list = evaluation(model_t, audio_s, video_s, fusion, dev_dataloader)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        print(f"dev_score : {dev_fbeta}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_epoch = epoch
            _SaveModel(fusion, save_path)

            fusion.eval()
            test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_dataloader)
            test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
            _, _, test_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
            test_acc = accuracy_score(test_label_list, test_pred_list)
            print(f"test_score (weighted-F1): {test_fbeta:.4f}  acc: {test_acc:.4f}  macro-F1: {test_macro:.4f}")
            best_metrics = {
                "best_epoch": best_epoch,
                "dev_weighted_f1": round(float(best_dev_fscore), 6),
                "test_accuracy": round(float(test_acc), 6),
                "test_weighted_f1": round(float(test_fbeta), 6),
                "test_macro_f1": round(float(test_macro), 6),
            }

    if results_path and best_metrics:
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({"baseline_TelME": best_metrics}, f, indent=2)
        print(f"Baseline metrics saved to {results_path}")

    return best_metrics

def evaluation(model_t, audio_s, video_s, fusion, dataloader):
    fusion.eval()
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.to(device), attention_masks.to(device), audio_inputs.to(device), video_inputs.to(device), batch_labels.to(device)
                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
                
            pred_logits = fusion(text_hidden, audio_hidden, video_hidden)
            
            """Calculation"""    

            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list


def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'total_fusion.bin'))

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3
    """Dataset Loading"""
    
    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(project_root, 'data', 'MELD.Raw') + os.sep

    train_path = data_path + 'train_meld_emo.csv'
    dev_path = data_path + 'dev_meld_emo.csv'
    test_path = data_path + 'test_meld_emo.csv'


    train_dataset = meld_dataset(preprocessing(train_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=2, collate_fn=make_batchs)


    dev_dataset = meld_dataset(preprocessing(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=2, collate_fn=make_batchs)

    test_dataset = meld_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=2, collate_fn=make_batchs)

    save_path = os.path.join(project_root, 'models', 'checkpoints')

    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    clsNum = len(train_dataset.emoList)
    init_config = Config()

    '''teacher model load'''
    teacher_ckpt = os.path.join(project_root, 'models', 'checkpoints', 'teacher.bin')
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.to(device)
    model_t.eval()

    '''student model'''
    audio_ckpt = os.path.join(project_root, 'models', 'checkpoints', 'student_audio', 'total_student.bin')
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load(audio_ckpt, map_location=device))
    for para in audio_s.parameters():
        para.requires_grad = False
    audio_s = audio_s.to(device)
    audio_s.eval()

    video_ckpt = os.path.join(project_root, 'models', 'checkpoints', 'student_video', 'total_student.bin')
    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load(video_ckpt, map_location=device))
    for para in video_s.parameters():
        para.requires_grad = False
    video_s = video_s.to(device)
    video_s.eval()

    '''fusion'''
    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion = fusion.to(device)
    fusion.eval()


    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    results_path = os.path.join(project_root, 'results', 'baseline.json')
    model_train(training_epochs, model_t, audio_s, video_s, fusion, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path, results_path=results_path)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    args = parse_args()
    main(args)
