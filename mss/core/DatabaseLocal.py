import copy

class DatabaseLocal:
    def __init__(self, column, wList): #column should be contain all of affecting parameter e.g. 'Genre', 'Key', 'Energy'. wList should contain data for each column, order according to column.
        self.mainArr = []
        self.CONST_MAP = column
        self.ref_song_list = []
        for song in wList[0]:
            self.ref_song_list.extend(song)
        for index in range(len(column)):
            all_genre = []
            for sub_album in wList[index]:
                curr_genre = []
                for song in sub_album:
                    curr_genre.append(self.ref_song_list.index(song))
                all_genre.append(curr_genre)
            self.mainArr.append(all_genre)
        self.__oriArr = copy.deepcopy(self.mainArr)

    def pop(self, val):
        resets = False
        for index in range(len(self.CONST_MAP)):
            for index2 in range(len(self.mainArr[index])):
                if(len(self.mainArr[index][index2])<1):
                    continue
                try:
                    self.mainArr[index][index2].remove(val)
                    if (len(self.mainArr[index][index2])<1):
                        resets = False
                except:
                    pass

        if resets:
            self.mainArr = copy.deepcopy(self.__oriArr) 

    def getList(self):
        return self.mainArr

    def getFreshList(self):
        return copy.deepcopy(self.__oriArr)

    def forceFlush(self):
        self.mainArr = copy.deepcopy(self.__oriArr) 
    
    def popVal(self, indexes,column=0, locSong=0):
        t = self.mainArr[column][indexes][locSong] #default: retrieve first value found
        self.pop(t)
        return t
    
    def __translate_name__ (self, idx):
        return self.ref_song_list[idx]

    def pushVal(self, wList):
        for song in wList[0]:
            self.ref_song_list.extend(song)
        for index in range(len(self.CONST_MAP)):
            for idx in range(len(wList[index])):
                for song in (wList[index][idx]):
                    self.mainArr[index][idx].extend([self.ref_song_list.index(song)])
                    self.__oriArr[index][idx].extend([self.ref_song_list.index(song)])