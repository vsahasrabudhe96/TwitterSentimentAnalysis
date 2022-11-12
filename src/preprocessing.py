import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocessing(df,col):
    
    # df = df.loc[:,df.columns.isin([col])]
    wl = WordNetLemmatizer()
    ## Removing the rows with links
    for i,val in enumerate(df[col]):
        if df[col][i].startswith(('http','www')):
            df.drop(index=[i],inplace=True)
    ## Remove usernames from tweets
    df[col] = df[col].apply(lambda s:re.sub('@[^\s]+','',s))
    ## Remove punctuations
    df[col] = df[col].apply(lambda s:s.translate(str.maketrans('', '', string.punctuation)))
    ## Remove Emojis
    
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    
    df[col] = df[col].apply(lambda x:emoj.sub("",x))
    
    ## Remove stop words
    filtered_sentence = []
    stop_words = set(stopwords.words('english'))
    for i in df[col]:
        if i not in stop_words:
            filtered_sentence.append(i)
    df[col] = filtered_sentence
    
    ## Tokenize the data
    tokenized = []
    for i in df[col]:
        tokenized.append(nltk.word_tokenize(i))
    df[col] = tokenized
    
    ## Lemmatize the tweets
    lemmatized = []
    le_token = []
    for i in df[col]: #List of words
        lemmatized_output = ' '.join([wl.lemmatize(w) for w in i])
        le_token_sent = [wl.lemmatize(w) for w in i]
        lemmatized.append(lemmatized_output)
        # le_token.append(le_token_sent)
    df[col] = lemmatized
    # df['token_lemma'] = le_token
    
    df.reset_index(drop=True,inplace=True)
    for i,val in enumerate(df[col]):
        if df[col][i] == "":
            df.drop(index=[i],inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    df[col] = df[col].apply(lambda x:x.lower())
    
    df = df.loc[:,df.columns.isin([col])]
    return df
    
    