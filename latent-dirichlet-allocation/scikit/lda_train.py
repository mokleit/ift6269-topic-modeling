import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

# This will load the preprocessed data and train an lda model that we will use in 'lda_save_topics'
# Set blei = True if you want to fit the model with respect to Blei's corpus. blei = False
# will use the speech recordings corpus.
# The lengh of the vocabulary can be set via the num_features variable

blei = True
num_features = 200

if blei:
    corpus_path = '../data/blei_samples.txt'
    vocabulary_path = 'blei/blei_vocabulary.csv'
    pickle_path = 'blei/lda_model.pkl'
else:
    corpus_path = 'speech/corpus.txt'
    vocabulary_path = 'speech/speech_vocabulary.csv'
    pickle_path = 'speech/lda_model.pkl'

# Load corpus
corpus = np.loadtxt(corpus_path, delimiter='\n', dtype=str)

# Transform corpus and define vocabulary
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features,
                                   stop_words='english', lowercase=True)
X = count_vectorizer.fit_transform(corpus)
vocabulary = count_vectorizer.get_feature_names()

# Save vocabulary
np.savetxt(vocabulary_path, vocabulary, delimiter=',', fmt='%s')

# Transform to pandas dataframe
df_count = pd.DataFrame(X.todense(), columns=vocabulary)

# Set up grid search
search_params = {
    'n_components': [10],
    'learning_decay': [.5],
    'learning_method': ['batch']
}

lda = LatentDirichletAllocation(max_iter=10, batch_size=24, n_jobs=-1)
model = GridSearchCV(lda, param_grid=search_params)

# Fit model
model.fit(X)

# Best model
best_model = model.best_estimator_

lda_pickle = pickle_path
with open(lda_pickle, 'wb') as file:
    pickle.dump(best_model, file)
