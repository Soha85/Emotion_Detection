import pandas as pd

Tweets = pd.read_csv('D:\\PHD Thesis\\2018-E-c-En-train\\2018-E-c-En-train.csv')
Tweets.info()
labels = list(Tweets.columns[2:])
print(labels)