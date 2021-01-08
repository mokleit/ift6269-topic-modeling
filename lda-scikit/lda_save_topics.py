import pickle
import numpy as np

# Run this in order to save the topics results
# This python file will load the model that was previously trained and saved in a pickle file.
# Set blei = True if you want the topics for Blei corpus. Otherwise, set it to false for the speech recordings.
# Set the n_top_words to be saved just below

blei = True
n_top_words = 10

if blei:
    vocabulary_path = 'blei/blei_vocabulary.csv'
    pickle_path = 'blei/lda_model.pkl'
    topics_path = 'blei/blei_topics.txt'
else:
    vocabulary_path = 'speech/speech_vocabulary.csv'
    pickle_path = 'speech/lda_model.pkl'
    topics_path = 'speech/speech_topics.txt'

with open(pickle_path, 'rb') as file:
    lda_model = pickle.load(file)

vocabulary = np.loadtxt(vocabulary_path, delimiter=',', dtype=str)
topic_words = {}

for topic, comp in enumerate(lda_model.components_):
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words[topic] = [vocabulary[i] for i in word_idx]

for topic, words in topic_words.items():
    top = 'Topic: %d' % topic
    top_words = '  %s' % ', '.join(words)
    output = open(topics_path, "a+")
    output.write(top + '\n')
    output.write(top_words + '\n')
    output.close()
