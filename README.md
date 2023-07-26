# Sentiment-Analysis-on-Recession-using-Tweets
import pandas as pd
import matplotlib.pyplot as plt
import nltk, re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
df= pd.read_csv("/content/recession_2023_india_tweets_dataset.csv")
df.head()
df=df.drop(['date', 'id', 'username', 'likeCount', 'retweetCount'],axis=1)
df.isnull().sum()

def clean(text):
  text=str(text).lower()
  text=re.sub(r'https?:\/\/\S+','',text)
  text=re.sub(r'[^a-zA-Z\s]','',text)
  return text

df['content']=df['content'].apply(clean)
df

pip install nltk
nltk.download('vader_lexicon')

for index, row in df['content'].iteritems():
  score= SentimentIntensityAnalyzer().polarity_scores(row)
  neg=score['neg']
  neu=score['neu']
  pos=score['pos']
  if neg>pos:
    df.loc[index, 'Sentiment'] = 'Negative'
  elif pos>neg:
    df.loc[index, 'Sentiment'] = 'Positive'
  else:
    df.loc[index, 'Sentiment'] = 'Neutral'

df

def count_tp_in_column(data,feature):
  total = data.loc[:,feature].value_counts(dropna=False)
  percentage = data.loc[:,feature].value_counts(dropna=False, normalize=True)*100
  return pd.concat([total,round(percentage,2)],axis=1,keys=['Total','Percentage'])

tp=count_tp_in_column(df,'Sentiment')
tp

labels = tp.index
sizes = tp['Percentage']
colors= [ '#ff9999', '#66b3ff', '#99ff99']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90)

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig=plt.gcf()
fig.gca().add_artist(centre_circle)

ax1.axis('equal')
plt.tight_layout()
plt.legend(title='Sentiment')
plt
all_tweets = " ".join(tweet for tweet in df['content'])

wordcloud = WordCloud(width=1024, height=512, random_state=21, max_font_size=110, background_color='white').generate(all_tweets)

plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt
all_positive_tweets = " ".join(tweet for tweet in df[df['Sentiment'] == 'Positive']['content'])

wordcloud=WordCloud(width=1024, height=512, random_state=21, max_font_size=110, background_color='white').generate(all_positive_tweets)

plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt
all_negative_tweets=" ".join(tweet for tweet in df[df['Sentiment'] == 'Negative']['content'])

wordcloud=WordCloud(width=1024, height=512, random_state=21, max_font_size=110, background_color='white').generate(all_negative_tweets)

plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt
