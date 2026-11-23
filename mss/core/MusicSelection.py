from .MusicSimilarity import MusicSimilarity
import numpy as np
class MusicSelection:
    def __init__(self, oneLoop):
        self.currentSong = 0
        self.currentEnergy = 0
        self.currentKey = 0
        self.playedSong = [[] for isi in range(5)]
        self.mSim = 0
        self.oneLoop = oneLoop

    def manualNextMusic(self, mstorage, songName):
        # try:
        flag_first = False
        if (self.currentSong == 0):
            flag_first = True
        self.currentSong = songName
        currentSongidx = mstorage.getData()['song_dir'].index(songName) 
        self.currentKey = mstorage.getData()['key'][currentSongidx]
        self.currentEnergy =  mstorage.getData()['energy'][currentSongidx]
        if (flag_first):
            energi = []
            a = self.currentEnergy + 1
            b = self.currentEnergy - 1
            if (a<=4):
                energi.append(a)
            if (b>=0):
                energi.append(b)
            energi.append(self.currentEnergy)
            candidateSongPool = []
            for isi in energi:
                candidateSongPool.extend([mstorage.getData()['song_dir'][i] for i, x in enumerate(mstorage.getData()['energy']) if (x == isi) and (currentSongidx != i)])
            self.mSim = MusicSimilarity(candidateSongPool,self.currentSong, mstorage)
        if (self.currentSong not in self.playedSong[self.currentEnergy]):
            self.playedSong[self.currentEnergy].append(self.currentSong)
        return True
    
    def getNext(self, mstorage, status=0):
        #phase one: check if there any song available
        count_played = 0
        for isi in self.playedSong:
            count_played+= len(isi)
        if ((len(mstorage.getData()['song_dir'])<= count_played) and (self.oneLoop)):
            print("Concert Finished!")
            return("Finish")
        elif (len(mstorage.getData()['song_dir'])<= count_played):
            self.playedSong = [[] for isi in range(5)]
        if (self.currentSong == 0):
            #first song output
            self.currentEnergy =  min(mstorage.getData()['energy'])
            currentSongidx = mstorage.getData()['energy'].index(min(mstorage.getData()['energy'])) #get lowest energy song
            self.currentSong = mstorage.getData()['song_dir'][currentSongidx]
            self.currentKey = mstorage.getData()['key'][currentSongidx]
            energi = []
            a = self.currentEnergy + 1
            b = self.currentEnergy - 1
            if (a<=4):
                energi.append(a)
            if (b>=0):
                energi.append(b)
            energi.append(self.currentEnergy)
            candidateSongPool = []
            for isi in energi:
                candidateSongPool.extend([mstorage.getData()['song_dir'][i] for i, x in enumerate(mstorage.getData()['energy']) if (x == isi) and (currentSongidx != i)])
            self.mSim = MusicSimilarity(candidateSongPool,self.currentSong, mstorage)
        else:
            prevEnergy = self.currentEnergy
            #phase 1: update energy level, repeat until there are available song list
            statusLoop = True
            candidateSongidx = [[] for isi in range(5)]
            for index in range(5):
                candidateSongidx[index] = [i for i, x in enumerate(mstorage.getData()['energy']) if (x == index)]
                candidateSongidx[index] = [cddIdx for cddIdx in candidateSongidx[index] if (not(mstorage.getData()['song_dir'][cddIdx] in self.playedSong[index]))]
            availableEnergy = [isi for isi in range(5) if (len(candidateSongidx[isi])>0)]
            if (len(availableEnergy)<=0):
                print("Concert Finished!")
                return("Finish")
            if (status==0):
                self.currentEnergy += 1
            elif (status==1):
                self.currentEnergy -= 1
            #phase 2: make sure the 0<=x<=4
            if (self.currentEnergy<0):
                self.currentEnergy = 0
            elif (self.currentEnergy>4):
                self.currentEnergy = 4
            if (status==0):
                tmp_energy = [isi for isi in availableEnergy if (isi>=self.currentEnergy)]
                if (len(tmp_energy)>0):
                    availableEnergy = tmp_energy
            elif (status==1):
                tmp_energy = [isi for isi in availableEnergy if (isi<=self.currentEnergy)]
                if (len(tmp_energy)>0):
                    availableEnergy = tmp_energy
            distAvailableEnergy = [abs(self.currentEnergy-NextEnergy) for NextEnergy in availableEnergy]
            self.currentEnergy = availableEnergy[np.argsort(distAvailableEnergy)[0]]
            candidateSongidx = candidateSongidx[self.currentEnergy]
            patience = 2
            statusLoop = True
            # phase 3: store songs with proper energy level, then filter based on the key condition. If none key exist, repeat until get desired
            candidateKey = self.__generate_key__(self.currentKey)
            stayCandidateSongidx = [cddIdx for cddIdx in candidateSongidx if (mstorage.getData()['key'][cddIdx]==self.currentKey)]
            draw_lots = np.random.choice([False, True], 1,p=[0.2,0.8]) #80% same key
            isStayCddIdxAv = len(stayCandidateSongidx)>0
            if (isStayCddIdxAv and np.random.choice([False, True], 1,p=[0.2,0.8])): #80% same key
                ncandidateSongidx = stayCandidateSongidx
                statusLoop = False
            while ((patience >=0) and (statusLoop)):
                ncandidateSongidx = [cddIdx for cddIdx in candidateSongidx if (mstorage.getData()['key'][cddIdx] in candidateKey)]
                if (len(ncandidateSongidx)>0):
                    statusLoop = False
                else:
                    if (patience>0):
                        candidateKeyNew = [self.__generate_key__(isi) for isi in candidateKey]
                        candidateKey = [isi for isi in candidateKey]
                        candidateKey.extend([isidalem for isi in candidateKeyNew for isidalem in isi])
                        candidateKey = set(candidateKey)
                    patience -= 1
            if (statusLoop): #if key still not found
                if (isStayCddIdxAv): #if same key still exist, do not change key
                    ncandidateSongidx = stayCandidateSongidx
                #don't filter as no choice
                else:
                    ncandidateSongidx = candidateSongidx
            # phase 5: check song similarity
            candidateSongPool = [mstorage.getData()['song_dir'][i] for i in ncandidateSongidx]
            currentSongidx = self.mSim.MusicSim(candidateSongPool, mstorage)
            currentSongidx =  mstorage.getData()['song_dir'].index(currentSongidx)
            self.currentSong = mstorage.getData()['song_dir'][currentSongidx]
            self.currentKey = mstorage.getData()['key'][currentSongidx]
        # phase 6: Output song
        self.playedSong[self.currentEnergy].append(self.currentSong)
        return self.currentSong

    def __generate_key__(self, currKey):
        pc = 12
        rel_pos = currKey % pc
        t1 = rel_pos +1
        t2 = rel_pos - 1
        t3 = rel_pos
        if (t2 < 0):
            t2 += pc 
        if (t1 >= pc):
            t1 = 0
        if (currKey >=  pc):
            t1 += pc 
            t2 += pc
        else:
            t3 += pc
        
        return (int(t1), int(t2), int(t3))
