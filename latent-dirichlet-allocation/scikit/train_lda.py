import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

num_features = 200

# Load corpus
corpus = np.loadtxt('corpus.txt', delimiter='\n', dtype=str)

# Transform corpus and define vocabulary
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features,
                                   stop_words='english', lowercase=True)
X = count_vectorizer.fit_transform(corpus)
vocabulary = count_vectorizer.get_feature_names()

# Save vocabulary
np.savetxt('speech_vocabulary.csv', vocabulary, delimiter=',', fmt='%s')

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

lda_pickle = 'lda_model.pkl'
with open(lda_pickle, 'wb') as file:
    pickle.dump(best_model, file)
