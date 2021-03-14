"""
Extract data from dataset
https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics
Select artists and their lyrics for clustering
Dump to JSON
"""

__author__ = 'don.tuggener@zhaw.ch'

import pandas   # run 'pip install pandas' if you don't have it
import json
from collections import defaultdict, Counter

# Input file
SONG_CSV = 'data/lyrics.csv'	# Download this file from https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics
# To see all genres: print(set(df['genre']))
GENRES = ['Indie', 'Electronic', 'Metal', 'Country', 'Jazz', 'Folk', 'Rock', 'Hip-Hop', 'Pop', 'R&B']

def select_tracks(n=10):
    """ 
    Per genre, get the tracks of the 10 artists with the most tracks 
    Also, remember genre of the artists
    """
    df = pandas.read_csv(SONG_CSV)
    df = df.fillna('')                              # Replace NaN elements with empty string
    selected_tracks = []
    artist2genre = {}
    for genre in GENRES:
        print(genre)
        tracks = df[df['genre'] == genre]           # Select all songs in the genre
        tracks = tracks[tracks['lyrics'] != '']     # Filter tracks with empty lyrics
        artist_tracks_count = Counter(list(tracks['artist']))
        selected_artists = artist_tracks_count.most_common(n)
        selected_artists = [a[0] for a in selected_artists]
        for a in selected_artists:
            artist2genre[a] = genre
        selected_tracks += [t for t in tracks.iterrows() if t[1]['artist'] in selected_artists]
    return selected_tracks, artist2genre

def lyrics_per_artist(artist_list, track_list):
    """ From the selected artists and tracks, merge all lyrics per artist """
    artist_lyrics = defaultdict(str)
    for artist, track in zip(artist_list, track_list):
        artist_lyrics[artist] += ' ' + track
    return artist_lyrics

if __name__ == '__main__':
    print('Getting tracks')
    selected_tracks, artist2genre = select_tracks()
    print('Merging lyrics per selected artist')
    artist_list = [t[1][3] for t in selected_tracks]
    track_list = [t[1][5] for t in selected_tracks]
    artist_lyrics = lyrics_per_artist(artist_list, track_list)
    json.dump(artist_lyrics, open('data/artist_lyrics_kaggle.json', 'w', encoding='utf-8'))
    json.dump(artist2genre, open('data/artist2genre_kaggle.json', 'w', encoding='utf-8'))