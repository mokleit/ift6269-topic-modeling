import pickle
import numpy as np

with open('lda_model.pkl', 'rb') as file:
    lda_model = pickle.load(file)

n_top_words = 10
vocabulary = np.loadtxt('speech_vocabulary.csv', delimiter=',', dtype=str)
topic_words = {}

for topic, comp in enumerate(lda_model.components_):
    word_idx = np.argsort(comp)[::-1][:n_top_words]
    topic_words[topic] = [vocabulary[i] for i in word_idx]

for topic, words in topic_words.items():
    top = 'Topic: %d' % topic
    top_words = '  %s' % ', '.join(words)
    print(top)
    print(top_words)
    output = open('speech_topics.txt', "a")
    output.write(top + '\n')
    output.write(top_words + '\n')
    output.close()
