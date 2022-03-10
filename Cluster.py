# import required sklearn libs
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# import other required libs
import pandas as pd
import numpy as np

# string manipulation libs
import re
import string
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba

np.random.seed(11)

df_true = pd.read_csv('True.csv',header = 0,encoding = "ISO-8859-1")
df_true_corpus = df_true["title"]
str_true = []
for i in range(0,len(df_true_corpus)):
    seg_list = jieba.cut(df_true_corpus.values[i])
    str_true.append(" ".join(seg_list))

pass
# initialize vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
X = vectorizer.fit_transform(str_true)
# initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=11)
kmeans.fit(X)
clusters = kmeans.labels_
pass

str_clusters0_true = ""
str_clusters1_true = ""
str_clusters2_true = ""

for i in range(0,len(str_true)):
    if (clusters[i] == 0):
        str = str_true[i]
        str_clusters0_true = str_clusters0_true + " " + str
    elif (clusters[i] == 1):
        str = str_true[i]
        str_clusters1_true = str_clusters1_true + " " + str
    else:
        str = str_true[i]
        str_clusters2_true = str_clusters2_true + " " + str

word_cloud = WordCloud(font_path="simsun.ttc",
                       background_color="white")
word_cloud.generate(str_clusters0_true)
plt.title('True News Cluster0')
plt.imshow(word_cloud)
plt.axis("off")
plt.show()

word_cloud = WordCloud(font_path="simsun.ttc",
                       background_color="white")
word_cloud.generate(str_clusters1_true)
plt.title('True News Cluster1')
plt.imshow(word_cloud)
plt.axis("off")
plt.show()

word_cloud = WordCloud(font_path="simsun.ttc",
                       background_color="white")
word_cloud.generate(str_clusters2_true)
plt.title('True News Cluster2')
plt.imshow(word_cloud)
plt.axis("off")
plt.show()



df_fake= pd.read_csv('Fake.csv',header = 0,encoding = "ISO-8859-1")
df_fake_corpus = df_fake["title"]
str_fake = []
for i in range(0,len(df_fake_corpus)):
    seg_list = jieba.cut(df_fake_corpus.values[i])
    str_fake.append(" ".join(seg_list))


# initialize vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
X = vectorizer.fit_transform(str_fake)
# initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=11)
kmeans.fit(X)
clusters = kmeans.labels_
pass

str_clusters0_fake = ""
str_clusters1_fake = ""
str_clusters2_fake = ""

for i in range(0,len(str_fake)):
    if (clusters[i] == 0):
        str = str_fake[i]
        str_clusters0_fake = str_clusters0_fake + " " + str
    elif (clusters[i] == 1):
        str = str_fake[i]
        str_clusters1_fake = str_clusters1_fake + " " + str
    else:
        str = str_fake[i]
        str_clusters2_fake = str_clusters2_fake + " " + str

word_cloud = WordCloud(font_path="simsun.ttc",
                       background_color="white")
word_cloud.generate(str_clusters0_fake)
plt.title('Fake News Cluster0')
plt.imshow(word_cloud)
plt.axis("off")
plt.show()

word_cloud = WordCloud(font_path="simsun.ttc",
                       background_color="white")
word_cloud.generate(str_clusters1_fake)
plt.title('Fake News Cluster1')
plt.imshow(word_cloud)
plt.axis("off")
plt.show()

word_cloud = WordCloud(font_path="simsun.ttc",
                       background_color="white")
word_cloud.generate(str_clusters2_fake)
plt.title('Fake News Cluster2')
plt.imshow(word_cloud)
plt.axis("off")
plt.show()