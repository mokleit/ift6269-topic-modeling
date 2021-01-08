# TOPIC MODELING USING LATENT DIRICHLET ALLOCATION (LDA)

## SCIKIT IMPLEMENTATION

We implement LDA using scikit-learn on two different datasets.

### Datasets

1. Text documents from the Associated Press found [here](http://www.cs.columbia.edu/~blei/lda-c/) and [here](lda-python/data/blei_samples.txt) in our project. 
2. Speech-to-Text recordings of IFT6269 lectures at the MILA (Université de Montréal) found [here](lda-scikit/speech/speech_recordings).

### Pre-processing

We pre-process the data and store it as [corpus.txt](lda-scikit/speech/corpus.txt).

### Training 

1. Use [train](lda-scikit/lda_train.py) to train the model and save it as a pickle file.
2. Use [save](lda-scikit/lda_save_topics.py) to save the topics extracted from training.

## PYTHON IMPLEMENTATION

### DATASETS

1. Text documents from the scribe notes of IFT6269 [here](lda-python/data/scribenotes)
2. Text documents from the Associated Press found [here](http://www.cs.columbia.edu/~blei/lda-c/) and [here](lda-python/data/blei_samples.txt) in our project. 

### Pre-processing and training

Completed in [lda](lda-python/lda.py). However, it needs to be fixed and cleaned as it was run in Colab.

