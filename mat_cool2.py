# This script is used with 3 different models (listed below). The script requests user input for a word of interest and
# performs the Word2Vec analysis. After using the 3 models, the results are plotted in a bar chart.
# Models were generated with:
# Model with Epoch = 30 (python phrase2vec.py --corpus=data/corpus_example --model_name=model_epoch_30 --epochs=30)
# Model with Epoch = 20 (python phrase2vec.py --corpus=data/corpus_example --model_name=model_epoch_20 --epochs=20)
# Model with Epoch = 40 (python phrase2vec.py --corpus=data/corpus_example --model_name=model_epoch_40 --epochs=40)

import sys, os
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import time

# List of models with different epochs
models = ['model_epoch_20', 'model_epoch_30', 'model_epoch_40']

# Request word of interest (woi)
woi = input('Enter word of interest: ')

for model in models:
    # Open model with different epochs
    w2v_model = Word2Vec.load(os.path.join(os.getcwd() + f'\\mat2vec\\training\\models\\{model}'))
    # Identify most similar words to entry
    words = w2v_model.wv.most_similar(woi)
    # For bar chart plotting
    word, score = zip(*words)
    word_x = [i for i, _ in enumerate(word)]
    plt.bar(word_x, score)
    plt.xlabel("Similar Words")
    plt.ylabel("Score")
    plt.title(f'Similar Words to {woi} with {model}')
    plt.xticks(word_x, word)
    plt.show()
