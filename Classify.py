import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import re
from emoji import UNICODE_EMOJI
import emoji
import string
# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
class Classify:
    def __init__(self):
        return
    def loadData(self):
        Tweets = pd.read_csv('2018-E-c-En-train.csv')
        labels = list(Tweets.columns[2:])
        return Tweets,labels
    def PreprocessData(self,Tweet):
        Tweet =  re.sub(r"http:\S+", '', Tweet)
        Tweet = emoji.demojize(Tweet)
        Tweet = re.sub(r'[' + string.punctuation + ']', '', Tweet)
        return Tweet
    def SplitHashTags(self,Tweet):
        return re.findall(r"#(\w+)",Tweet)
    def Emdedding(self,Tweet):
        inputs = tokenizer(Tweet, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        return embeddings