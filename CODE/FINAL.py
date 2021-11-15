### This code does the entire processing of the proposed system ###

from keras.preprocessing.sequence import pad_sequences
import csv
import re
from html.parser import HTMLParser
import preprocessor as p
from keras import backend as K
from sys import argv
K.set_image_data_format('channels_last')
import numpy as np
import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import pandas as pd
import pickle
from keras.models import load_model
import en_core_web_sm

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Encountered a start tag:", tag)

    def handle_endtag(self, tag):
        print("Encountered an end tag :", tag)

    def handle_data(self, data):
        print("Encountered some data  :", data)


def preprocess(tweets):
    preprocessed=[]
    for tweet in tweets:
        #print(tweet)
        tweet = re.sub('RT @[\w_]+: ', '', tweet)
        tweet = tweet.lower()
        # Leaving out url, emoji, @mentions, RT
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', tweet)
        p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.MENTION,p.OPT.SMILEY)
        tweet=p.clean(tweet)
        # Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        # replaces &amp like chars with original symbols
        html_parser=MyHTMLParser()
        tweet=html_parser.unescape(tweet)
        # Changing apostrophe like're to are
        apostrophe={"let's":"let us","'s":" is","'re":" are","n't":" not","'d":" would","'m":" am","'ve":" have","'ll":" will"}
        words=tweet.split()
        #print(words)
        reformed=[]
        for word in words:
            for a in apostrophe:
                if(word.find(a)!=-1):
                    word=word.replace(a,apostrophe[a])
                    #print(word)
            reformed.append(word)
        tweet=" ".join(reformed)
        # trim
        tweet = tweet.strip('\'"')
        # remove non ascii characters
        tweet = re.sub(r'[^ a-zA-Z0-9]+', r'', tweet)
        preprocessed.append(tweet)
    return preprocessed

if __name__ == "__main__":

    print("folder name:"+argv[1])
    path='C:\\Users\\Swars\\Desktop\\FYP\\'+argv[1]+'\\'
    print(path)

    ###     TEXT PROCESSING     ###

    model = load_model('Data//lstm.h5')
    with open('Data\\tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #Assuming data is of format tweetid,tweettext
    data = pd.read_csv(path+'input.csv')
    data['tweet_text']=preprocess(data['tweet_text'])
    X=data['tweet_text']
    sequence = tokenizer.texts_to_sequences(X)
    seq = pad_sequences(sequence, maxlen=30)
    Y=model.predict(seq)
    output=[]
    for result in Y:
        if result>0.7:
            output.append("offer")
        elif result<0.3:
            output.append("request")
        else:
            output.append("none")
    op=pd.DataFrame(output)
    data['result']=op
    data.to_csv(path+'After_text_prediction.csv')

    ###     IMAGE PROCESSING     ###

    model = load_model('Data/mymodel_vgg.h5')
    data_path=path+'Images//'
    img_list = os.listdir(data_path)
    if(len(img_list)!=0):
        img_rows = 224
        img_cols = 224
        count=0
        img_data_list=[]
        for img in img_list:
              img_path = data_path+img
              img = image.load_img(img_path, target_size=(224, 224))
              x = image.img_to_array(img)
              x = np.expand_dims(x, axis=0)
              x = preprocess_input(x)
              x = x/255
              count+=1
              if(count==101):
                  break
              img_data_list.append(x)

        img_data = np.array(img_data_list)
        img_data = img_data.astype('float32')
        img_data=np.rollaxis(img_data,1,0)
        img_data=img_data[0]
        X=img_data[:101]
        Y=model.predict(X)
        op_class=np.argmax(Y,axis=1)


        tweetid=[]
        count=0
        for name in img_list:
            tweetid.append(name.split('_')[0])
            count+=1
            if(count==101):
                break

        df = pd.DataFrame({'tweet_id' : tweetid,'severity' : op_class}, columns=['tweet_id','severity'])
        df.to_csv(path+'After_image_prediction.csv')

        Reader = csv.reader(open(path+'After_image_prediction.csv'))
        result = {}

        for row in Reader:
            idx = row[1]
            values = row[2]
            if idx in result:
                if values < result[idx]:
                    result[idx] = values
            else:
                result[idx] = values

        with open('C:\\Users\\Swars\\Desktop\\FYP\\FINAL\\After_image_prediction.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in result.items():
               writer.writerow([key, value])

        f1 = pd.read_csv(path+'After_text_prediction.csv')
        f2 = pd.read_csv(path+'After_image_prediction.csv')
        f = pd.merge(left=f1, right=f2, how='left', on='tweet_id')
        f.fillna(2,inplace=True)
        path1=path+'Final.csv'
        f.to_csv(path1)
    else:
        data['severity']=0
        path1=path + 'After_text_prediction.csv'
        data.to_csv(path1)

    ###     MATCHING     ###

    nlp = en_core_web_sm.load()
    data = pd.read_csv(path1)

    df1 = data[data['result'] == 'request']
    df2 = data[data['result'] == 'offer']

    print('\n')
    print('-' * 500)
    print("TWEET_ID\t\t\t|\tTWEET_TEXT")
    print('-' * 500)

    #For each offer searching a request
    for index1, offer in df2.iterrows():
        target = nlp(offer.tweet_text, "utf-8")
        print(offer.tweet_id,'\t|\t',offer.tweet_text)
        maxm = 0.80
        index2 = 0
        match = pd.Series([])

        for index2, request in df1.iterrows():
            sent = nlp(request.tweet_text)
            measure = target.similarity(sent) #similarity score between offer and request
            if measure > maxm:
                if match.empty:
                    match = request
                    maxm = measure
                else:
                    if measure - maxm < 0.01:
                        if match.severity < request.severity:
                            match = request
                            maxm = measure
                    else:
                        match = request
                        maxm = measure
        if match.empty:
            print("*** NO MATCH ***")
        else:
            index2 = df2[df2['tweet_id'] == match.tweet_id].index
            df2.drop(index2, inplace=True)
            print(match.tweet_id, '\t|\t', match.tweet_text)
            print(' ' * 18, '\t|\t', 'MATCHING SIMILARITY : ' + str(maxm))
        print('-' * 500)