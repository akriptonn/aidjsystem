from .Storage import Storage

class MusicStorage(Storage):
    def __init__(self, location, columns, saveEveryAdd = True):
        super(MusicStorage, self).__init__(location, columns, saveEveryAdd)

    def getFeature(self, column, song):
        idx = self.getData()['song_dir'].index(song)
        temp = {}
        columnList = []
        if isinstance(column, list):
            columnList.extend(column)
        else:
            columnList.append(column)
        for isi in columnList:
            temp[isi] = self.getData()[isi][idx]
        return temp

    
