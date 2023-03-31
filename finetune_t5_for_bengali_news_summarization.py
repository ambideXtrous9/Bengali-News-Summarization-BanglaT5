import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW, T5ForConditionalGeneration,AutoTokenizer as Tokenizer)

pl.seed_everything (42)

fname = 'bengali_train.jsonl'
with open(fname) as f:
    data = [json.loads(line) for line in f]

# Convert the list of dictionaries to a Pandas dataframe
train_df = pd.DataFrame(data)

# Print the dataframe
train_df = train_df[['title','summary','text']]
train_df

train_df['len_summary'] = train_df['summary'].apply(lambda x: len(x.split()))
train_df['len_text'] = train_df['text'].apply(lambda x: len(x.split()))

train_df

max_index = train_df['len_summary'].idxmax()
sample = train_df.iloc[max_index]
sample['summary']

sample['len_summary']

min_index = train_df['len_summary'].idxmin()
sample = train_df.iloc[min_index]
sample['summary']

sample['len_summary']

# Plot histogram of length column
plt.hist(train_df['len_summary'], bins=30)
# Set x and y labels
plt.xlabel('Length of Summary')
plt.ylabel('Count')

# Show the plot
plt.show()

# Plot histogram of length column
plt.hist(train_df['len_text'], bins=30)
# Set x and y labels
plt.xlabel('Length of Text')
plt.ylabel('Count')

# Show the plot
plt.show()

num_short_summaries = (train_df['len_summary'] <= 3).sum()
num_short_summaries

train_df = train_df[train_df['len_summary'] > 3]

train_df.shape

# Plot histogram of length column
plt.hist(train_df['len_summary'], bins=30)
# Set x and y labels
plt.xlabel('Length of Summary')
plt.ylabel('Count')

# Show the plot
plt.show()

# Plot histogram of length column
plt.hist(train_df['len_text'], bins=30)
# Set x and y labels
plt.xlabel('Length of Summary')
plt.ylabel('Count')

# Show the plot
plt.show()

"""##**Val Data**"""

fname = 'bengali_val.jsonl'
with open(fname) as f:
    data = [json.loads(line) for line in f]

# Convert the list of dictionaries to a Pandas dataframe
val_df = pd.DataFrame(data)

# Print the dataframe
val_df = val_df[['title','summary','text']]
val_df

val_df['len_summary'] = val_df['summary'].apply(lambda x: len(x.split()))
val_df['len_text'] = val_df['text'].apply(lambda x: len(x.split()))

val_df

max_index = val_df['len_summary'].idxmax()
sample = val_df.iloc[max_index]
sample['summary']

sample['len_summary']

min_index = val_df['len_summary'].idxmin()
sample = val_df.iloc[min_index]
sample['summary']

sample['len_summary']

# Plot histogram of length column
plt.hist(val_df['len_summary'], bins=30)
# Set x and y labels
plt.xlabel('Length of Summary')
plt.ylabel('Count')

# Show the plot
plt.show()

# Plot histogram of length column
plt.hist(val_df['len_text'], bins=30)
# Set x and y labels
plt.xlabel('Length of Text')
plt.ylabel('Count')

# Show the plot
plt.show()

"""## **Test Data**"""

fname = 'bengali_test.jsonl'
with open(fname) as f:
    data = [json.loads(line) for line in f]

# Convert the list of dictionaries to a Pandas dataframe
test_df = pd.DataFrame(data)

# Print the dataframe
test_df = test_df[['title','summary','text']]
test_df

test_df['len_summary'] = test_df['summary'].apply(lambda x: len(x.split()))
test_df['len_text'] = test_df['text'].apply(lambda x: len(x.split()))

test_df

max_index = test_df['len_summary'].idxmax()
sample = test_df.iloc[max_index]
sample['summary']

sample['len_summary']

min_index = test_df['len_summary'].idxmin()
sample = test_df.iloc[min_index]
sample['summary']

sample['len_summary']

# Plot histogram of length column
plt.hist(test_df['len_summary'], bins=30)
# Set x and y labels
plt.xlabel('Length of Summary')
plt.ylabel('Count')

# Show the plot
plt.show()

# Plot histogram of length column
plt.hist(test_df['len_text'], bins=30)
# Set x and y labels
plt.xlabel('Length of Text')
plt.ylabel('Count')

# Show the plot
plt.show()

"""## **Model and Tokenizer**"""

MODEL_NAME = 'flax-community/bengali-t5-base'

tokenizer = Tokenizer.from_pretrained(MODEL_NAME)
LModel = T5ForConditionalGeneration.from_pretrained(MODEL_NAME,return_dict=True)


"""## **DataLoader**"""

class BengaliDataset(Dataset):
  def __init__(self,data : pd.DataFrame,tokenizer : Tokenizer,source_max_token_len : int = 512,
               target_max_token_len : int = 64):

    self.tokenizer = tokenizer
    self.data = data
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,index : int):
    data_row = self.data.iloc[index]
    text = data_row['text']
    summary = data_row['summary']
    

    source_encoding = tokenizer(
        text,
        max_length = self.source_max_token_len,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = "pt")
    
    target_encoding = tokenizer(
        summary,
        max_length = self.target_max_token_len,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = "pt")
    
    labels = target_encoding["input_ids"]
    labels[labels == 0] = -100

    return dict(
        text_input_ids = source_encoding['input_ids'].flatten(),
        text_attention_mask = source_encoding['attention_mask'].flatten(),
        labels = labels.flatten(),
        labels_attention_mask = target_encoding["attention_mask"].flatten())

class BengaliDataModule(pl.LightningDataModule):
  def __init__(self,train_df : pd.DataFrame,val_df : pd.DataFrame,test_df : pd.DataFrame,
               tokenizer : Tokenizer,batch_size : int = 8,source_max_token_len : int = 512,
               target_max_token_len : int = 64):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.val_df = val_df
    self.tokenizer = tokenizer
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def setup(self,stage=None):
    self.train_dataset = BengaliDataset(self.train_df,self.tokenizer,self.source_max_token_len,self.target_max_token_len)
    self.val_dataset = BengaliDataset(self.val_df,self.tokenizer,self.source_max_token_len,self.target_max_token_len)
    self.test_dataset = BengaliDataset(self.test_df,self.tokenizer,self.source_max_token_len,self.target_max_token_len)

  def train_dataloader(self):
    return DataLoader(self.train_dataset,batch_size = self.batch_size,shuffle=True,num_workers=4)

  def val_dataloader(self):
    return DataLoader(self.val_dataset,batch_size = self.batch_size,num_workers=4)

  def test_dataloader(self):
    return DataLoader(self.test_dataset,batch_size = self.batch_size,num_workers=4)

class BengaliSummaryModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = LModel
    
    def forward(self,input_ids,attention_mask,decoder_attention_mask, labels=None):
        
        output = self.model(
            input_ids,
            attention_mask = attention_mask,
            labels = labels,
            decoder_attention_mask = decoder_attention_mask
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

BATCH_SIZE = 16
N_EPOCHS = 10

data_module = BengaliDataModule(train_df,val_df,test_df,tokenizer,batch_size = BATCH_SIZE)
data_module.setup()

model = BengaliSummaryModel()

checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = 'best_cp',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_loss',
    mode = 'min'
)

trainer = pl.Trainer(gpus = 1,
    callbacks=[checkpoint_callback],
    max_epochs = N_EPOCHS,
)

trainer.fit(model,data_module)

trained_model = BengaliSummaryModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
trained_model.freeze()


trainer.test(trained_model, data_module)


def summarize_text(text):
    text_encoding = tokenizer(
        text,
        max_length = 512,
        padding = "max_length",
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = "pt"
    )
    
    generated_ids = trained_model.model.generate(
        input_ids = text_encoding["input_ids"],
        attention_mask = text_encoding["attention_mask"],
        max_length=32,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    
    preds = [tokenizer.decode(gen_id,skip_special_tokens=True,clean_up_tokenization_spaces=True)
             for gen_id in generated_ids]
    
    return " ".join(preds)


sample_row = test_df.iloc[0]
text = sample_row["text"]
model_summary = summarize_text(text)

print("Actual Text : ",text)
print('----------------------------------------------------------')
print("Actual : ",sample_row["summary"])
print("----------------------------------------------------------")
print("Predicted : ",model_summary)