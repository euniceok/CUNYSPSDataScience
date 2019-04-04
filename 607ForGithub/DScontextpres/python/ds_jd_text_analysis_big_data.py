
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')




# references: https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/

filepath = "/Users/euniceok/PycharmProjects/cuny/spring2019/Week8/Data-607-Project-3"
outputfilepath = '/Users/euniceok/PycharmProjects/cuny/spring2019/Week8/Data-607-Project-3/Eunice/output/cleandata'
filename = '/source/all_jobs.csv'

jddf = pd.read_csv(filepath + filename).drop('Unnamed: 0', axis=1)
jddf = jddf.drop_duplicates()

# desc = jddf['job_description'].str.cat(sep=' ')

jddf['tokens'] = jddf['job_description'].apply(lambda x: nltk.word_tokenize(x))
jddf['sent_tokens'] = jddf['job_description'].apply(lambda x: nltk.sent_tokenize(x))

# create list of stopwords to filter out of list of tokens
stop_words = set(stopwords.words("english"))

# remove stopwords
jddf['filt_tokens'] = np.nan
jddf['filt_tokens'] = jddf['tokens'].apply(lambda x: [item for item in x if item not in stop_words])

# lemmatize words
lem = WordNetLemmatizer()

jddf['lemm_tokens'] = jddf['filt_tokens'].apply(lambda x: [lem.lemmatize(item) for item in x]) # defaults to assuming the part of speech is noun

# join clean words back into sentences to assign polarity scores
separator = ' '
jddf['clean_desc'] = jddf['lemm_tokens'].map(lambda x: separator.join(x))

sia = SIA()
results = []

# for jd in jddf['job_description']:
for jd in jddf['clean_desc']:
    pol_score = sia.polarity_scores(jd)
    pol_score['jd'] = jd
    results.append(pol_score)

df = pd.DataFrame.from_records(results)

full = pd.merge(df, jddf, how='left', left_on='jd', right_on='clean_desc').drop('clean_desc',axis=1)

full['label'] = 0
full.loc[full['compound'] > 0.2, 'label'] = 1
full.loc[full['compound'] < -0.2, 'label'] = -1

full['city'] = full['job_location'].apply(lambda x: x.split(',')[0].strip())
full['state'] = full['job_location'].apply(lambda x: x.split(',')[-1][:3].strip())

full.to_csv('/Users/euniceok/PycharmProjects/cuny/spring2019/Week8/Data-607-Project-3/Eunice/big_data_sent_scored_cln.csv')


# read csv back in
ind = pd.read_csv(filepath + '/Eunice/big_data_sent_scored_cln.csv').drop('Unnamed: 0', axis=1)

ind.shape

# shape data for plot of distribution of sentiments
sentiment = ind.label.value_counts(normalize=True) * 100
sentiment.loc[0] = 0
sentiment.loc[-1] = 0

fig, ax = plt.subplots(figsize=(8, 8))
counts = sentiment
sns.barplot(x=counts.index, y=counts, ax=ax, color = 'b')
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")
ax.set_title('Overall Sentiment Distribution')
plt.savefig(outputfilepath + '/sentdistro.png')

# because the sentiments are all positive, let's take a look at how positive
mostpos = ind.sort_values('pos', ascending = False).head(10)
leastpos = ind.sort_values('pos', ascending = False).tail(10)
# see jd samples of the most and least positive


# bar plot of city by average pos sentiment
city = ind.groupby(['city'])['compound'].mean().sort_values(ascending=False)

city_top10 = city.head(10)
city_bot10 = city.tail(10)

fig, ax = plt.subplots(figsize=(4, 8))
counts = city_top10
sns.barplot(x=counts, y=counts.index, ax=ax, color='b')
# ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_xlabel("Sentiment Score")
ax.set_ylabel("City")
ax.set_title('Distribution of Sentiment Across Top 10 City')
#plt.xticks(rotation=90)
plt.savefig(outputfilepath + '/top_city.png',bbox_inches='tight')

fig, ax = plt.subplots(figsize=(4, 8))
counts = city_bot10
sns.barplot(x=counts, y=counts.index, ax=ax, color='b')
# ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_xlabel("Sentiment Score")
ax.set_ylabel("City")
ax.set_title('Distribution of Sentiment Across Bottom 10 City')
#plt.xticks(rotation=90)
plt.savefig(outputfilepath + '/bot_city.png',bbox_inches='tight')


# bar plot of state by average pos sentiment
state = ind.groupby(['state'])['compound'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(4, 8))
counts = state
sns.barplot(x=counts, y=counts.index, ax=ax, color='b')
# ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_xlabel("Sentiment Score")
ax.set_ylabel("State")
ax.set_title('Distribution of Sentiment Across State')
#plt.xticks(rotation=90)
plt.savefig(outputfilepath + '/state.png',bbox_inches='tight')

# bar plot of count of ds jd by state
state_ct = ind.groupby(['state'])['state'].count().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(4, 8))
counts = state_ct
sns.barplot(x=counts, y=counts.index, ax=ax, color='b')
# ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_xlabel("Count of DS JDs")
ax.set_ylabel("State")
ax.set_title('Count of Data Scientist Job Descriptions In Each State')
#plt.xticks(rotation=90)
plt.savefig(outputfilepath + '/state_ct.png',bbox_inches='tight')


# # bar plot of industry by average pos sentiment
# industry = ind.groupby(['industry'])['compound'].mean().sort_values(ascending=False)
# fig, ax = plt.subplots(figsize=(8, 8))
# counts = industry
# sns.barplot(x=counts, y=counts.index, ax=ax, color='b')
# # ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
# ax.set_xlabel("Sentiment Score")
# ax.set_ylabel("Industry")
# ax.set_title('Distribution of Sentiment Across Industry')
# #plt.xticks(rotation=90)
# plt.savefig(outputfilepath + '/industry.png',bbox_inches='tight')