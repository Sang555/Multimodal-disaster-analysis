#This module retrieves the trending tweets. Checks if they are related to any disaster.
# If yes, the live tweets are retrieved and stored for further processing.
import tweepy
import json
import csv
import wget
import FINAL
import os
import subprocess

consumer_key = 'nnFQmlkLS3xsotwI38gLIy3tZ'
consumer_secret = 'yuZsgb0TLz70eFak1pUZMZjDKowQxlXYWxAEPoahWQ4EbpqCYp'

access_token = '935191682043088896-Cci5SSiuGvE0uA8mXjICbDvc7dUo8n1'
access_token_secret = 'gpuzWb36V3R1c1hkl0iemYJ2ioUxJyITRw22yz8i8fkyl'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

#Returns the current trending topics in India
trend_topics=[]
INDIA_WOE_ID = 2282863
india_trends = api.trends_place(INDIA_WOE_ID)
trends = json.loads(json.dumps(india_trends, indent=1))
print("-----------------TRENDING TOPICS IN INDIA-----------------")
for trend in trends[0]["trends"]:
    trend_topics.append((trend["name"].strip("#")).lower())
    print(trend["name"])
print("---------------------------------------------------\n")

#Check if they are related to diaster
flag=0
possible_disasters=['avalanche','volcano','earthquake','tsunami',
'tornado','drought','storm','blizzard','flood','tremor','disaster']
for topic in trend_topics:
    for d in possible_disasters:
        if(topic.find(d)!=-1):
            flag=1
            disaster=topic
            print(topic)
            print("**** A Disaster has occured !!! ****\n")
            break

#Retrieve all the tweets related to the particular disaster
tweets={}
if(flag!=0):
    count=10
    csvFile = open('C:\\Users\\Swars\\Desktop\\FYP\\LIVE\\input.csv', 'w',encoding='utf8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(["tweet_id","tweet_text"])
    for tweet in tweepy.Cursor(api.search, q="#"+disaster, tweet_mode="extended",lang="en",place=INDIA_WOE_ID).items():
        if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
            tweets[tweet.id]=tweet.entities
            print(tweet.created_at, tweet.id, tweet.full_text,tweet.entities)
            csvWriter.writerow([tweet.id, tweet.full_text.encode('utf-8').decode('utf8')])
        count = count - 1
        if(count<0):
            break
    #Checking if the tweet contains image. If yes, storing the media content.
    count=0
    for id,entities in tweets.items():
        media = tweet.entities.get('media', [])
        number=len(media)
        if (len(media) > 0):
            i=0
            while(i<number):
                media_file = media[i]['media_url']
                count = count + 1
                wget.download(media_file,'C:\\Users\\Swars\\Desktop\\FYP\\LIVE\\Images').rename('C:\\Users\\Swars\\Desktop\\FYP\\LIVE\\images'+id+"_"+i)
                i=i+1
    csvFile.close()
    ###   PROCESSING  ###
    os.system("python FINAL.py LIVE")
