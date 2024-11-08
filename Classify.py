import pandas as pd
class Classify:
    def __init__(self):
        return
    def loadData(self):
        try:
            Tweets = pd.read_csv('2018-E-c-En-train.csv')
            labels = list(Tweets.columns[2:])
            return Tweets,labels
        except Exception as e:
            return e