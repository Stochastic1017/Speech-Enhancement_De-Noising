
import librosa
import noisereduce as nr
import numpy as np
import pandas as pd
import time
import glob
import os
import shutil
import soundfile as sf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def statistics(y):
    
    mean = np.mean(y)
    std = np.std(y) 
    maxv = np.amax(y) 
    minv = np.amin(y) 
    skew = np.sum((y - mean)**3) / (len(y) * std**3) if std != 0 else 0
    q1 = np.quantile(y, 0.25)
    q3 = np.quantile(y, 0.75)
    
    return np.array([mean, 
                     maxv, 
                     minv, 
                     std, 
                     q1, 
                     q3, 
                     skew])

def spectral(y, feature):
    
    if feature == 'orig':
        return y
    
    elif feature == 'rms':
        return librosa.feature.rms(y=y)[0]
    
    elif feature == 'zcr':
        return librosa.feature.zero_crossing_rate(y=y)[0]
    
    if feature == 'spec_centroid':
        return librosa.feature.spectral_centroid(y=y)[0]
    
    elif feature == 'spec_bandwidth':
        return librosa.feature.spectral_bandwidth(y=y)[0]
    
    elif feature == 'spec_contrast':
        return librosa.feature.spectral_contrast(y=y)[0]
    
    if feature == 'spec_flatness':
        return librosa.feature.spectral_flatness(y=y)[0]
    
    elif feature == 'spec_rolloff':
        return librosa.feature.spectral_rolloff(y=y)[0]
    
    elif feature == 'mfcc':
        return librosa.feature.mfcc(y=y)[0]
    
    elif feature == 'poly_features':
        return librosa.feature.poly_features(y=y)[0]
    
    elif feature == 'melspectrogram':
        return librosa.feature.melspectrogram(y=y)[0]
    
    elif feature == 'chroma_stft':
        return librosa.feature.chroma_stft(y=y)[0]    
        
    elif feature == 'chroma_cqt':
        return librosa.feature.chroma_cqt(y=y)[0]
    
    elif feature == 'chroma_cens':
        return librosa.feature.chroma_cens(y=y)[0]
    
    else:
        print('Invalid feature, choose one from:')
        print('orig', 
              '\nrms', 
              '\nzcr', 
              '\nspec_centroid', 
              '\nspec_contrast', 
              '\nspec_flatness', 
              '\nspec_rolloff', 
              '\nmfcc', 
              '\npoly_features', 
              '\nmelspectrogram',
              '\nchroma_stft',
              '\nchroma_cens')

def feature_stacking(y, features):
    
    temp = []  
    
    for feature in features:
        curr = statistics(spectral(y, feature = feature))
        temp.append(curr)
        
    return np.vstack(temp)

def speech_detection(y, window_size = 10000, stride = 100, threshold = .025):
    moving_avg = []
    for i in range(0, len(y) - window_size + 1, stride):
        window = y[i:i + window_size]
        moving_avg.append(np.mean(window))
    if (np.abs(np.max(moving_avg) - np.min(moving_avg)) * 100) < threshold:
        return False
    else: 
        return True

def clean_audios(df, language):
    
    os.makedirs(f'{language}_cleaned', exist_ok = True)
    source_directory = f'{language}'
    destination_directory = f'{language}_cleaned'
    count_ignore = 0
    count_fix = 0
    count_untouched = 0
    
    for index in range(len(df)):
        if df['outlier'][index] == 0:
            file_name = df['filename'][index].replace(f"{language}/", "")
            print(f"{index} :: {file_name} not an outlier! Copying and moving to {destination_directory}.")      
            source_path = os.path.join(source_directory, file_name)
            destination_path = os.path.join(destination_directory, file_name)
            shutil.copy(source_path, destination_path)
            count_untouched += 1
        else:
            y, sr = librosa.load(df['filename'][index])
            if speech_detection(y) == False:
                file_name = df['filename'][index].replace(f"{language}/", "")
                print(f"{index} :: {file_name} not a fixable outlier!, moving onto next file.")
                count_ignore += 1
                continue
            else:
                file_name = df['filename'][index].replace(f"{language}/", "")
                print(f"{index} :: {file_name} is a fixable outlier! Fixing and writing it to {destination_directory}.")    
                reduced_noise = nr.reduce_noise(y = y, sr = sr)
                file_path = os.path.join(destination_directory, file_name)
                sf.write(file_path, reduced_noise, samplerate = sr)
                count_fix += 1
    
    print('#########################################') 
    print('#########################################')        
    print('Number of files that were too bad to fix:', count_ignore)
    print('Number of files that were fixable:', count_fix)
    print('Number of files that were left untouched:', count_untouched)
    print('#########################################') 
    print('#########################################') 
    
features = ['orig',
            'rms', 
            'zcr',
            'spec_flatness', 
            'mfcc',
            'chroma_stft']

language='belarusian_1'

count = 0
files_thrown = []
temp = []

for files in glob.glob(f'{language}/*.mp3'):
    if count % 100 == 0:
        print('Currently at:', files, '->', count, 'out of 1500')   
    try:
        y, sr = librosa.load(files)
    except Exception as e:
        print(f"\nError loading audio file: {e}")
        print(f"Skipping {files} due to an issue with feature stacking.\n")
        files_thrown.append(files)
        continue

    temp.append(feature_stacking(y, features))
    count += 1
    
X = np.stack(temp).reshape(np.stack(temp).shape[0], -1)

scaler = StandardScaler().fit(X)
pca = PCA(n_components=3)
components = pca.fit_transform(scaler.transform(X))

clf = IsolationForest(random_state = None, contamination = 0.05).fit(components)
predictions = clf.predict(components)
IF_indices = np.where(predictions == -1)[0]
print(IF_indices)

clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.05)
predictions = clf.fit_predict(components)
LOF_indices = np.where(predictions == -1)[0]
print(LOF_indices)

print('Anomalies detected:\n', np.intersect1d(LOF_indices, IF_indices))

outliers = np.zeros(len(components[:, 0]))
outliers[np.intersect1d(LOF_indices, IF_indices)] = 1
all_files = glob.glob(f'{language}/*.mp3')
for files in files_thrown:
    if files in all_files:
        all_files.remove(files)
        
df = pd.DataFrame({'filename': all_files,
                   'PC1': components[:, 0], 
                   'PC2': components[:, 1], 
                   'PC3': components[:, 2], 
                   'outlier': outliers})
df = df.reset_index(drop = False)

print(df)

clean_audios(df = df, language = language)
