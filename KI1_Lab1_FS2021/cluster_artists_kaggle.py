"""
Cluster artists based on the words in their lyrics
Data from kaggle
Format data with create_data_kaggle.py
"""

__author__ = 'don.tuggener@zhaw.ch'

import numpy
import json
import pdb
import re
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def plot_dendrogram(clustered, artists):
    """ Plot a dendrogram from the hierarchical clustering of the artist lyrics """
    #plt.figure(figsize=(25, 10))   # for orientation = 'bottom'|'top'
    plt.figure(figsize=(10, 25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')  # this' but the label of the whole axis!
    plt.ylabel('Artists')
    plt.tight_layout()
    dendrogram(clustered,
        #leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels = artists,
        orientation = 'left',
    )
    #plt.show() # Instead pf saving
    plt.savefig('example_dendrogram.svg', bbox_inches='tight')  

def words_per_artist(artist_lyrics, lyrics_tfidf_matrix, ix2word, n=10):
    """ 
    For each artist, print the most highly weighted words acc. to TF IDF 
    Print n words that are above the mean weight 
    """
    
    # transform matrix to dataframe -> easier to work with
    df = pd.DataFrame(lyrics_tfidf_matrix.todense())
    
    for idx, artist in enumerate(artist_lyrics.keys()):
        
        # get words with rating of artist
        words_of_artist = df.iloc[idx]
        
        word_dict = {}
        
        # iterate over words and filter non zero
        for wordIdx, value in enumerate(words_of_artist):
            if value != 0:
                word_dict[wordIdx] = value
                
        # mean weight
        mean_weight = sum(word_dict.values()) / len(word_dict)
        
        
        # sort dict after tfidf, descending and choose top ten
        top_ten_words = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True)[:n])
        
        
        # print top ten words
        for wordIdx in top_ten_words:
            # change word from idx to word
            word = ix2word[wordIdx]
            tf_idf = top_ten_words[wordIdx]
            
            # check if over mean weight
            if tf_idf > mean_weight:
                print(artist, word, tf_idf)
             
def getIx2Word(words):
    ix2word = {}
    for idx,word in enumerate(words):
        ix2word[idx] = word
    return ix2word
                 

if __name__ == '__main__':
    
    print('Loading data')
    artist2genre = json.load(open('data/artist2genre_kaggle.json', 'r', encoding='utf-8'))
    artist_lyrics = json.load(open('data/artist_lyrics_kaggle.json', 'r', encoding='utf-8'))
    # Custom tokenization to remove numbers etc.
    lyrics = [' '.join(re.findall('[A-Za-z]+', l)) for l in artist_lyrics.values()]
    
    print('Vectorizing with TF IDF')
    # Vectorize the song lyrics
    # TODO implement; create lyrics_tfidf_matrix (artist/word matrix) and ix2word (dict that maps word IDs to words)
    vectorizer = TfidfVectorizer(stop_words = 'english')
    lyrics_tfidf_matrix = vectorizer.fit_transform(lyrics)
    ix2word = getIx2Word(vectorizer.get_feature_names())
    
    print('Distinct words per artist')
    words_per_artist(artist_lyrics, lyrics_tfidf_matrix, ix2word)

    print('Clustering')
    clustered = KMeans(n_clusters=10, n_init=50).fit(lyrics_tfidf_matrix)
    order_centers = clustered.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(10):
        print("Cluster %d" % i)
        for ind in order_centers[i, :30]:
            print(';%s' % terms[ind])
        print("\n")

    print('Plotting')
    artist_names = [a+': '+artist2genre[a].upper() for a in list(artist_lyrics.keys())]
    #plot_dendrogram(clustered, artist_names)
    plot_dendrogram(linkage(lyrics_tfidf_matrix.toarray(), method='ward'), artist_names)
