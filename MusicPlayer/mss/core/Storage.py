import json

class Storage:
    def __init__(self, location, columns, saveEveryAdd = True):
        self.location = location
        self.data = {}
        self._list_column_ = []
        self.addColumn(columns)
        self.saveEveryAdd = saveEveryAdd
        self.loadStorage()

    def getData(self):
        return self.data
        
    def addColumn(self, columns):
        if (isinstance(columns, list)):
            for content in columns:
                self.__add_column__(content)
        else:
            self.__add_column__(columns)

    def __add_column__(self,content):
        if (content in self._list_column_):
            pass
        else:
            self._list_column_.append(content)
            self.data[content] = []
        
    def addData(self,data):
        for column, content in data.items():
            if column in self.data:
                if (isinstance(content, list)):
                    self.data[column].extend(content)
                else:
                    self.data[column].append(content)
        if (self.saveEveryAdd):
            self.saveData()
        
    def saveData(self):
        self.getData()
        with open(self.location, 'w') as fp:
            json.dump(self.data, fp, indent=4) 
    
    def removeColumn(self, columns):
        target_column = []
        if (isinstance(columns, list)):
            target_column.extend(columns)
        else:
            target_column.append(columns)
        for content in target_column:
            if (content in self._list_column_):
                self.data[content] = []
                self._list_column_.remove(content)
        self.__refresh_storage()

    def __refresh_storage(self):
        temp = {}
        for column, content in self.data.items():
            if column in self._list_column_:
                temp[column] = content
        self.data = temp
    
    def resetStorage(self):
        self.data = {}
        self._list_column_ = []
    
    def loadStorage(self):
        try:
            with open(self.location, 'r') as fp:
                self.data = json.load(fp)
            for column in self.data:
                self.addColumn(column)
        except IOError as error:
            pass