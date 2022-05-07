import twint        #This is our main library. Gives us access to Twitter back end data.
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from textblob import TextBlob    #Allows us to assess the polarity of a string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#configuration
config = twint.Config()
config.Pandas = True             #allows pandas integration
config.Search = "#COP26"         #we are searching for this string across twitter with the following parameters
os.remove("demo_data.json")      #removes old json file when we change parameters
config.Min_likes = 20
config.Lang = "en"
config.Since = "2021-10-27 00:00:00"
config.Limit = 3200
config.Store_json = True
config.Output = "demo_data.json"        #outputs a json file with all tweet parameters such that we can manipulate it later
#running search
twint.run.Search(config)            #running search

tweet_list = []
with open("demo_data.json", "r", encoding="utf-8", errors='ignore') as json_file_input:
    temp_list = json_file_input.readlines()

    for line in temp_list:
        temp_dict = json.loads(line)
        tweet=temp_dict["tweet"]
        likes=temp_dict["likes_count"]
        replies=temp_dict["replies_count"]
        retweets=temp_dict["retweets_count"]
        video=temp_dict["video"]
        tweet_list.append({"tweet": tweet, "likes": likes, "replies": replies, "retweets": retweets})

df_likes_sub = pd.DataFrame(columns = ['Likes', 'Polarity', 'Replies', 'Retweets'])
for tweet_dict in tweet_list:
    tweet = tweet_dict["tweet"]
    likes = tweet_dict["likes"]
    replies = tweet_dict["replies"]
    retweets = tweet_dict["retweets"]

    tweet_blobbed = TextBlob(tweet)
    result = tweet_blobbed.sentiment
    polarity = result.polarity
    subjectivity = result.subjectivity
    adjusted_polarity = polarity*subjectivity
    if adjusted_polarity == 0:
        continue

    df_likes_sub=df_likes_sub.append({'Likes': likes, 'Replies': replies, 'Retweets': retweets, 'Polarity': adjusted_polarity}, ignore_index=True)

x1 = df_likes_sub["Likes"]
y = df_likes_sub["Polarity"]
plt.subplot(2, 2, 1)
plt.scatter(x1, y, marker=None)
plt.xlabel('Likes')
plt.ylabel('Adjusted Polarity')


x2 = df_likes_sub["Replies"]
plt.subplot(2, 2, 2)
plt.scatter(x2, y, marker=None)
plt.xlabel('Replies')


x3 = df_likes_sub["Retweets"]
plt.subplot(2, 2, 3)
plt.scatter(x3, y, marker=None)
plt.xlabel('Retweets')               #this last

plt.show()



text = (str(tweet_list[0:len(tweet_list)]))

stopwords = set(STOPWORDS)
stopwords.update(["https", "t", "co", "retweets", "tweet", "retweets'", "tweet'", "replies'", "likes'", "U000e0067", "U000e0074", "U000e0007f", "U000e0063", "U000e0062", "U000e0073", "de", "la", "U000e007f", "COP26"])
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()







