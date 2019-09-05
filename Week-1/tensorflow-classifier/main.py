import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *


from nltk.sentiment.vander import SentimentIntensityAnalyzer
import unicodedata



#Getting the data for twitter to perform sentiment analisys
def getDataFromTwitter(start_date, end_date, company):
    pass

#Getting the finance data of the company for Yahoo
def getDataPricesDataForYahoo(start_date, end_date, company):
    pass

#Performing sentiment analysis using linear regression
def analyseWithLinearRegression(data):

    #Read data
    df_stocks = 1
    
    #Convert from float to int
    df_stocks['prices'] = df_stocks['adj_close'].apply(np.int64)
    df_stocks = df_stocks[['prices','articles']]
    df_stocks['tweets'] = df_stocks['tweets'].map(lambda x:x.lstrip('.-'))

    df = df_stocks[['prices']].copy()

    #This variables comes from the sentiment analyser
    df['compound'] = ''
    df['neg'] = ''
    df['neu'] = ''
    df['pos'] = ''


    sid = SentimentIntensityAnalyzer

    for date, row in df_stocks.T.iteritems():
        try:
            #Get the tweet for that day formatting it
            sentence = unicodedata.normalize('NFKD', df_stocks.loc[date,'tweets']).encode('ascii','ignore')

            #Apply sentiment analisys using SentimentIntensityAnalyzer
            ss = sid.polarity_scores(sentence)

            #Fill those data to the array
            df.set_value(data,'compound', ss['compound'])
            df.set_value(data,'neg', ss['neg'])
            df.set_value(data,'neu', ss['neu'])
            df.set_value(data,'pos', ss['pos'])
        except TypeError:
            print df.df_stocks.loc[date, 'tweets']
            print date
        


    pass


#Ploting the data
