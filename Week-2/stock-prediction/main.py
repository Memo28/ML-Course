import numpy as np
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import tweepy
from sklearn.linear_model import LogisticRegression



from nltk.sentiment.vader import SentimentIntensityAnalyzer

import unicodedata
from datetime import datetime, timedelta


#---------------------------------------------------------------------------

consumer_key = 'PxfiletfKmIKyZrHxvBPTBmdP'
consumer_secret = 'M5vTz37wzEqrDbvLSHqDGqa7VJdaLLiy9GK1eS9gaxhUQyoOqc'

access_token = '324606030-OE05Hn9vaIruvNcufY2w7ZtwWd3MY5xwKDOuNAfR'
access_token_secret = 'qoQnAl641ISal5ji4IvjWpP1UlHWdnE2RChhcLZBozrW1'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#-------------------------------------------------------------------------


def offset_value(test_start_date, test, predictions_df):
    temp_date = test_start_date
    average_last_5_days_test = 0
    average_upcoming_5_days_predicted = 0
    total_days = 10
    for i in range(total_days):
        average_last_5_days_test += test.loc[temp_date, 'prices']
        temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
        difference = temp_date + timedelta(days=1)
        temp_date = difference.strftime('%Y-%m-%d')
    average_last_5_days_test = average_last_5_days_test / total_days

    temp_date = test_start_date
    for i in range(total_days):
        average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
        temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
        difference = temp_date + timedelta(days=1)
        temp_date = difference.strftime('%Y-%m-%d')
    average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
    difference_test_predicted_prices = average_last_5_days_test - average_upcoming_5_days_predicted
    return difference_test_predicted_prices

#Getting the data for twitter to perform sentiment analisys
def getDataFromTwitter(start_date, end_date, company):
    pass

#Getting the finance data of the company for Yahoo
def getDataPricesDataForYahoo(start_date, end_date, company):
    pass

#Performing sentiment analysis using linear regression
def analyseWithLinearRegression():

    #Read data

    # Reading the saved data pickle file
    df_stocks = pd.read_pickle('pickled_ten_year_filtered_data.pkl')    
    
    #Convert from float to int
    df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
    df_stocks = df_stocks[['prices','articles']]
    df_stocks['articles'] = df_stocks['articles'].map(lambda x:x.lstrip('.-'))

    df = df_stocks[['prices']].copy()

    #This variables comes from the sentiment analyser
    df['compound'] = ''
    df['neg'] = ''
    df['neu'] = ''
    df['pos'] = ''


    sid = SentimentIntensityAnalyzer()

    for date, row in df_stocks.T.iteritems():
        try:
            #Get the tweet for that day formatting it
            # sentence = unicodedata.normalize('NFKD', df_stocks.loc[date,'articles']).encode('ascii','ignore')
            sentence = df_stocks.loc[date,'articles']
            #Apply sentiment analisys using SentimentIntensityAnalyzer
            ss = sid.polarity_scores(sentence)

            #Fill those data to the array
            df.set_value(date,'compound', ss['compound'])
            df.set_value(date,'neg', ss['neg'])
            df.set_value(date,'neu', ss['neu'])
            df.set_value(date,'pos', ss['pos'])
        except TypeError:
            print (df.df_stocks.loc[date, 'articles'])
            print (date)

#Ploting the data
    # train_start_date = '2019-01-01'
    # train_end_date = '2019-02-05'

    # test_start_date = '2019-02-08'
    # test_end_date = '2019-02-15'

    train_start_date = '2007-01-01'
    train_end_date = '2014-12-31'
    test_start_date = '2015-01-01'
    test_end_date = '2016-12-31'


    years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    prediction_list = []

    for year in years:
    # Splitting the training and testing data
        train_start_date = str(year) + '-01-01'
        train_end_date = str(year) + '-10-31'
        test_start_date = str(year) + '-11-01'
        test_end_date = str(year) + '-12-31'

        #Dividing the data frame in test and training
        train = df.ix[train_start_date : train_end_date]
        test = df.ix[test_start_date:test_end_date]

        sentiment_score_list_train = []
        for date, row in train.T.iteritems():
            # sentiment_score = np.asarray(df.loc[date,'compound'], df.loc[date,'neg'] , df.loc[date,'neu'], df.loc[date,'pos'])
            sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])

            sentiment_score_list_train.append(sentiment_score)
        
        numpy_df_train = np.asarray(sentiment_score_list_train)

        sentiment_score_list_test = []
        for date, row in test.T.iteritems():
            # sentiment_score = np.asarray(df.loc[date,'compound'], df.loc[date,'neg'] , df.loc[date,'neu'], df.loc[date,'pos'])
            sentiment_score = np.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])

            
            sentiment_score_list_test.append(sentiment_score)
        
        numpy_df_test = np.asarray(sentiment_score_list_test)

        lr = LogisticRegression()
        lr.fit(numpy_df_train,train['prices'])

        prediction = lr.predict(numpy_df_test)
        prediction_list.append(prediction)

        idx = pd.date_range(test_start_date,test_end_date)

        predictions_df_list = pd.DataFrame(data=prediction[0:],index=idx,columns=['prices'])

        difference_test_predicted_prices = offset_value(test_start_date,test,predictions_df_list)

        predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices

        predictions_df_list['ewma'] = pd.ewm(predictions_df_list['prices'], span=10, frenq='D').mean()
        predictions_df_list['actual_value'] = test['prices']
        predictions_df_list['actual_value_ewma'] = pd.ewm(predictions_df_list['actual_value'], span=10, freq='D').mean()
        # Changing column names
        predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
        predictions_df_list.plot()


def run():
    analyseWithLinearRegression()

    


if __name__ == '__main__':
    run()