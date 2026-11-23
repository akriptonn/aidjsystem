import numpy as np
import pickle
from scipy.spatial.distance import euclidean as euclidean_distance

class MusicSimilarity():
    def __init__(self, songPool, firstSong, featureData):
        with open('tmp/pca.pkl', 'rb') as fp:
            self.tfer = pickle.load(fp)
        self.currentSongSD = self.__compute_feature__(firstSong,featureData)
        self.previousSongSD = self.currentSongSD
        song_themes = []
        for song in songPool:
            song_themes.append(self.__compute_feature__(song, featureData))
        self.centroid = np.average(np.array(song_themes),axis=0)


    def __compute_feature__(self, song, featureData):
        candidatefeature = ['melspectogram']
        f1Song = featureData.getFeature(candidatefeature, song)['melspectogram'][0]
        f1Song = np.array(f1Song)
        specCtrstAvgs = np.average(f1Song, axis=0)
        specValleyAvgs = np.std(f1Song, axis=0)
        def calculateDeltas(array):
            D = array[1:] - array[:-1]
            return D
        specCtrstDeltas = np.average(np.abs(calculateDeltas(f1Song)), axis=0)
        specValleyDeltas = np.std(np.abs(calculateDeltas(f1Song)), axis=0)
        return self.tfer.transform(np.average(np.array([specCtrstAvgs, specValleyAvgs, specCtrstDeltas, specValleyDeltas]),axis=1).reshape(1, -1))

    def MusicSim(self, candidateSong, featureData):
        cur_theme_centroid = 0.4 * self.centroid + 0.6 * (-0.1*self.previousSongSD + 1.1*self.currentSongSD)
        self.previousSongSD = self.currentSongSD
        song_options_distance_to_centroid = []
        for song in candidateSong:
            dist_to_centroid = euclidean_distance(cur_theme_centroid, self.__compute_feature__(song, featureData))
            song_options_distance_to_centroid.append(dist_to_centroid)
        song_options_closest_to_centroid = np.argsort(song_options_distance_to_centroid)
        self.currentSongSD = self.__compute_feature__(candidateSong[song_options_closest_to_centroid[0]], featureData)
        return candidateSong[song_options_closest_to_centroid[0]]

    