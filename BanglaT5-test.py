import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from normalizer import normalize

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW, AutoModelForSeq2SeqLM,AutoTokenizer as Tokenizer)


MODEL_NAME = 'csebuetnlp/banglat5'

tokenizer = Tokenizer.from_pretrained(MODEL_NAME,use_fast=False)
LModel = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,return_dict=True)

"""## **DataLoader**"""

class BengaliSummaryModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = LModel
    
    def forward(self,input_ids,attention_mask,decoder_attention_mask, labels=None):
        
        output = self.model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
        
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
        )
        
        self.log("train_loss",loss,prog_bar=True,logger=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
        )
        
        self.log("val_loss",loss,prog_bar=True,logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        
        loss, outputs = self(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = labels_attention_mask,
            labels = labels
        )
        
        self.log("test_loss",loss,prog_bar=True,logger=True)
        
        return loss
    
    
    def configure_optimizers(self):
        return AdamW(self.parameters(),lr = 0.0001)



cppath = 'BanglaT5.ckpt'
trained_model = BengaliSummaryModel.load_from_checkpoint(cppath)
trained_model.freeze()


def summarize_text(text):
    text_encoding = tokenizer(
        normalize(text),
        max_length = 512,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        return_tensors = "pt")
    
    generated_ids = trained_model.model.generate(
        input_ids = text_encoding["input_ids"],
        max_length=100,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    
    preds = [tokenizer.decode(gen_id,skip_special_tokens=True,clean_up_tokenization_spaces=True)
             for gen_id in generated_ids]
    
    return " ".join(preds)


with open('input.txt', 'r') as f:
    text = f.read()

model_summary = summarize_text(text)

with open('output.txt', 'a') as f:
    f.write('\n==========================================================\n')
    f.write("Actual Text : {}\n".format(text))
    f.write('----------------------------------------------------------\n')
    #f.write("Actual : {}\n".format(sample_row["summary"]))
    #f.write("----------------------------------------------------------\n")
    f.write("Predicted : {}\n".format(model_summary))
    f.write('==========================================================\n')


