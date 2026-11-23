from ..core import MusicLoader
import json
import os
import numpy as np

def exportCache(songdir, dirs = 'tmp/preshadered.bin', time=30):
    mLoader = MusicLoader.MusicLoader(time)
    t_d = mLoader.retrieveDataset(songdir, ignore_main_path=False)
    __saveShader__(t_d, dirs)

def __saveShader__(nwdata, dirc):
    if (True):
        data = nwdata
        for idx in range(len(data['songs_dir'])):
            data['songs_dir'][idx] = data['songs_dir'][idx].split('/')[-1]
        with open(dirc, 'w') as fp:
            json.dump(data, fp, indent=4)

def loadCache(dirc, songs_path, ignore_main_path=False, format='.mp3'):
    return __loadShader__(dirc,songs_path,ignore_main_path,format)

def __loadShader__(dirc, songs_path, ignore_main_path=False, format='.mp3'):
    newdata = {
        'mfcc' : [],
        'songs_dir': []
    }
    unloaded = []
    with open(dirc, 'r') as fp:
        data = json.load(fp)

    #isfile
    print(songs_path)
    if (os.path.isfile(songs_path)):
        if songs_path.lower().endswith(format):
            file_path = songs_path
            temp = file_path.split('/')[-1]
            if (temp in data['songs_dir']):
                t_a = np.array(data['songs_dir'])
                t_a = np.where(t_a == temp)[0]
                for idx in t_a: 
                    newdata['mfcc'].append(data['mfcc'][idx])
                    newdata['songs_dir'].append(file_path)
            else:
                print('missing '+temp)
                unloaded.append(file_path)
    #isdir
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(songs_path)):
        if ((dirpath not in songs_path) or (not ignore_main_path)):
            for f in filenames:
                file_path = os.path.join(dirpath,f)
                if f.lower().endswith(format):
                    temp = f
                    if (temp in data['songs_dir']):
                        t_a = np.array(data['songs_dir'])
                        t_a = np.where(t_a == temp)[0]
                        for idx in t_a: 
                            newdata['mfcc'].append(data['mfcc'][idx])
                            newdata['songs_dir'].append(file_path)
                    else:
                        print('missing '+ temp)
                        unloaded.append(file_path)
    
    return newdata, unloaded
