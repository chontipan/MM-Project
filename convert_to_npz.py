#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


AUDIO_DIR = 'fma_small'


# In[3]:


def get_tids_from_directory(audio_dir):
    """Get track IDs from the mp3s in a directory.
    Parameters
    ----------
    audio_dir : str
        Path to the directory where the audio files are stored.
    Returns
    -------
        A list of track IDs.
    """
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files)
    return tids


def get_audio_path(audio_dir, track_id):
    """
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.
    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


# In[4]:


tids = get_tids_from_directory(AUDIO_DIR)
print(len(tids))


# ### Function to create spectograms

# In[5]:


def create_spectogram(track_id):
    filename = get_audio_path(AUDIO_DIR, track_id)
    y, sr = librosa.load(filename)
    #print(y.shape)
    "Split songs into 3 slices. Each has 10 second"
    y= y[:660000] 
    song = []
    
    song.append(y[:220000])
    song.append(y[220000:440000])
    song.append(y[440000:])
    spec=[]
    for s in song:
        spect = librosa.feature.melspectrogram(y=s, sr=sr,n_fft=2048, hop_length=512)
        spect = librosa.power_to_db(spect, ref=np.max)
        #spect=spect.T
        spec.append(spect)
    return np.array(spec)


# In[6]:


def plot_spect(track_id):
    spect = create_spectogram(track_id)
    #print(spect.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect.T, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


# ### Load dataset with genre and track IDs

# In[8]:


filepath = 'csv/tracks.csv'
tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])
keep_cols = [('set', 'split'),
('set', 'subset'),('track', 'genre_top')]

df_all = tracks[keep_cols]
df_all = df_all[df_all[('set', 'subset')] == 'small']

df_all['track_id'] = df_all.index
df_all.head()


# In[9]:


df_all.shape


# In[10]:


df_all[('track', 'genre_top')].unique()


# In[11]:


dict_genres = {'Electronic':0,  'Folk':1,  'Pop' :2, 'Instrumental':3 }


# ### Create Arrays

# In[38]:


def create_array(df):
    genres = []
    
    X_spect = np.empty((0, 420, 128))
    count = 0
    #Code skips records in case of errors
    for index, row in df.iterrows():
        try:
            count += 1
            track_id = int(row['track_id'])
            genre = str(row[('track', 'genre_top')])
            
            
            spect = create_spectogram(track_id)
            #print(spect)
            for spec in spect:
                spect=spec.reshape(430,128)              
                spect = spect[:420, :] 
                X_spect = np.append(X_spect, [spect], axis=0)
                genres.append(dict_genres[genre])
            if count % 100 == 0:
                print("Currently processing: ", count)
        except:
            print("Couldn't process: ", count,"row['track_id']")
            continue
    y_arr = np.array(genres)
    return X_spect, y_arr


# In[13]:


df_all[('set', 'split')].unique()

"Take 4 genres out"

Rock = df_all[df_all[('track', 'genre_top')] == 'Rock'].index
International = df_all[df_all[('track', 'genre_top')] == 'International'].index
Experimental = df_all[df_all[('track', 'genre_top')] == 'Experimental'].index
Hip = df_all[df_all[('track', 'genre_top')] == 'Hip-Hop'].index
 
# Delete these row indexes from dataFrame
df_all.drop(Rock , inplace=True)
df_all.drop(International , inplace=True)
df_all.drop(Experimental , inplace=True)
df_all.drop(Hip , inplace=True)

# ### Create train, validation and test subsets

# In[14]:

df_train = df_all[df_all[('set', 'split')]=='training']
df_valid = df_all[df_all[('set', 'split')]=='validation']
df_test = df_all[df_all[('set', 'split')]=='test']

print(df_train.shape, df_valid.shape, df_test.shape)


# In[26]:

X_test, y_test = create_array(df_test)

np.savez('test_arr', X_test, y_test)

X_valid, y_valid = create_array(df_valid)

np.savez('valid_arr', X_valid, y_valid)

X_train, y_train = create_array(df_train)
np.savez('train_arr', X_train, y_train)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

X_train, y_train = unison_shuffled_copies(X_train, y_train)
X_valid, y_valid = unison_shuffled_copies(X_valid, y_valid)

np.savez('suf_train_arr', X_train, y_train)
np.savez('suf_valid_arr', X_valid, y_valid)
