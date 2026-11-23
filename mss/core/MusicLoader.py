import os
import librosa
import math
from sklearn.preprocessing import StandardScaler
import numpy as np

class MusicLoader:
    def __init__(self, duration=30,sr=22050, num_segments=10, n_mfcc=13, n_fft=4084, hop_length=1024):
        self.SAMPLES_PER_TRACK = sr*duration
        self.num_samples_per_segment = int(self.SAMPLES_PER_TRACK / num_segments) 
        self.expected_num_mfcc_vectors_per_segment = math.ceil(self.num_samples_per_segment / hop_length)
        self.num_segments = num_segments
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
    
    def extract_feature(self, file_path, format='.mp3'):
            
        signal, dumps = 0, 0
        try:
            signal, dumps = librosa.load(file_path)
        except:
            return [[False]]
            
        return_data = [] #num_of_segment, melspec, spectral_contrast

        for s in range(self.num_segments): 
            start_sample = self.num_samples_per_segment * s   
            finish_sample = self.num_samples_per_segment + start_sample
            try:
                mfcc = librosa.feature.mfcc(signal[start_sample : finish_sample],
                                                        sr = self.sr,
                                                        n_fft = self.n_fft,
                                                        n_mfcc = self.n_mfcc,
                                                        hop_length = self.hop_length)

                mfcc = mfcc.T

                melspec = librosa.feature.melspectrogram(signal[start_sample : finish_sample],
                                                    sr = self.sr,
                                                    n_fft = self.n_fft,
                                                    hop_length = self.hop_length)

                melspec = librosa.power_to_db(melspec, ref=np.max) #log scaling
                
                scalers = StandardScaler()
                melspec = scalers.fit_transform(melspec)

                melspec = melspec.T

                avgs = np.mean(melspec, dtype=np.float64)
                melspec -= avgs
                tonnetz = librosa.feature.tonnetz(signal[start_sample : finish_sample],
                                                        sr = self.sr,
                                                        hop_length = self.hop_length)
                tonnetz = tonnetz.T

            except:
                continue
                    # store mfcc for segment if it has the expected length
            if len(mfcc) == self.expected_num_mfcc_vectors_per_segment:
                return_data.append([True, melspec.tolist(), mfcc.tolist(), tonnetz.tolist()])
        
        return return_data



    def retrieveDataset(self, songs_path, ignore_main_path=False, format='.mp3'):
            # create container for the feature
        data = {
                'mfcc' : [],
                'song_dir': [],
                'melspectogram': [],
                'tonnetz': []
        }
            
        songs_path_list = []
        if (os.path.isfile(songs_path)):
            songs_path_list.append(songs_path)
        else:
            for i, (dirpath, dirnames, filenames) in enumerate(os.walk(songs_path)):
                if ((dirpath not in songs_path) or (not ignore_main_path)):
                        
                    dirpath_components = dirpath.split('/')
                    semantic_label = dirpath_components[-1]
                    for f in filenames:
                        file_path = os.path.join(dirpath,f)
                        songs_path_list.append(file_path)
            
        for file_path in songs_path_list:
            target_segment = []
            target_segment2 = []
            target_segment3 = []

            for segment in self.extract_feature(file_path, format):
                if (segment[0]):
                    target_segment.append(segment[1])
                    target_segment2.append(segment[2])
                    target_segment3.append(segment[3])
            if (len(target_segment)<=0):
                continue
            if (len(target_segment2)<=0):
                continue
            if (len(target_segment3)<=0):
                continue    
            data['song_dir'].append(file_path)
            data['melspectogram'].append(target_segment)
            data['mfcc'].append(target_segment2)
            data['tonnetz'].append(target_segment3)
            
        return data
