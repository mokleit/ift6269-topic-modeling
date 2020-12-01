import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import time


#folllowed this turotial: https://colab.research.google.com/drive/195tBon2Su1nLOg8YLBHYUrAsRC8ZIjQA#scrollTo=XY1E5-ngC0af

#inputs; 
n_features = 1000 #build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
n_components = 10 #Number of topics.
n_top_words = 20 #for prints
max_treshold = 0.95 # high frequency words
min_treshold = 2 #low frequency words


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

t0 = time()

###import dataset
df = []

with open('Sample_Blei.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split( r'\n')]
        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        df.append(inner_list[0])
        



# vectorize the documents 
print("Extracting tf-idf features for NMF...")
tf_vectorizer = CountVectorizer(max_df=max_treshold, min_df=min_treshold,
                                   max_features=n_features,
                                   stop_words='english')


tf = tf_vectorizer.fit_transform(df)


#run LDA
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)

print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)