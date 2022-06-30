#Author : Hao-Wei Huang, Muhammad Fadli, Achmad Kripton Nugraha
#AI DJ Project
from re import A, split
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QUrl,QThread
from madmom.models import BEATS_LSTM
from pydub import AudioSegment,effects
from math import ceil
from msvcrt import getch
from yodel.filter import Biquad
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Dropout

import cv2
import os
import tempfile
import pandas as pd
import ast
import pyrubberband as pyrb
import numpy as np
import madmom
import playergui,sys
import mss
import time
import numpy
import pyaudio
import subprocess
import threading
import scipy
import array
import mediapipe as mp

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
    def keyPressEvent(self, event):
        pressed_button = event.key()
        if pressed_button == QtCore.Qt.Key_Escape:
            self.close()
        else:
            pass
                
class MusicPlayer():
    def __init__(self):
        #Build User Interface Using PyQT5 
        self.app = QtWidgets.QApplication(sys.argv)
        self.SplashScreen = MainWindow()
        self.ui = playergui.Ui_SplashScreen()
        self.ui.setupUi(self.SplashScreen)
        self.SplashScreen.setWindowFlags(QtCore.Qt.FramelessWindowHint) # Remove title bar
        self.SplashScreen.setAttribute(QtCore.Qt.WA_TranslucentBackground) # Set background to transparent
        self.SplashScreen.show()
        
        #Value initialization
        self.vid = cv2.VideoCapture(0) #use the default webcam
        self.sliderModified = False
        self.pyaudioInitiated = False
        self.nowPlayingIndex = 0
        self.nextSongIndex = 0
        self.currentPlayer = 0
        self.chunkPos1 = 0
        self.chunkPos2 = 0
        self.equalizerFlag1 = 0
        self.equalizerFlag2 = 0
        self.time1 = 0 
        self.time2 = 0
        self.pauseState1 = True
        self.pauseState2 = True
        self.playedAlready1 = False
        self.playedAlready2 = False
        self.bpm1Changed = False
        self.bpm2Changed = False
        self.aligned1 = True
        self.aligned2 = True
        self.index = 0
        self.songIndex1 = 0
        self.songIndex2 = 0
        self.playlist = [[]]
        self.ui.volume1.setRange(-50, 0)
        self.ui.volume1.setValue(0)
        self.ui.volume2.setRange(-50, 0)
        self.ui.volume2.setValue(0)
        self.volume1 = 0
        self.volume2 = 0
        self.ui.songLengthSlider1.setValue(0)
        self.ui.songLengthSlider2.setValue(0)
        self.speedChangeFlag = False

        #Equalizer Value Set 
        self.sampleRate = 44100
        self.lowCutoff = 20
        self.midCenter = 1000
        self.highCutoff = 13000
        self.Q = 1.0 / np.sqrt(2)
        self.bquad_filter = Biquad()
        
        #Event listener
        self.ui.addFile.clicked.connect(self.addFile)
        self.ui.play1.clicked.connect(self.playSong1)
        self.ui.stop1.clicked.connect(self.stopSong1)
        self.ui.backward1.clicked.connect(self.rewindSong1)
        self.ui.forward1.clicked.connect(self.nextSong1)
        self.ui.volume1.valueChanged.connect(self.setVolume1)
        self.ui.songLengthSlider1.sliderPressed.connect(self.songSlider1Pressed)
        
        self.ui.play2.clicked.connect(self.playSong2)
        self.ui.stop2.clicked.connect(self.stopSong2)
        self.ui.backward2.clicked.connect(self.rewindSong2)
        self.ui.forward2.clicked.connect(self.nextSong2)
        self.ui.volume2.valueChanged.connect(self.setVolume2)
        self.ui.songLengthSlider2.sliderPressed.connect(self.songSlider2Pressed)
        self.qCrowd = []
        self.thread1 = threading.Thread(target=self.getNextMusic)

        #Run music selection system
        self.ms = mss.MusicSelectionSystem(".\\config\\mss.json", notloopPlaylist=True)

        #Execute User Interface
        sys.exit(self.app.exec_())

    def change_audioseg_tempo(self, audiosegment, tempo, new_tempo):
        y = np.array(audiosegment.get_array_of_samples())
        if audiosegment.channels == 2:
            y = y.reshape((-1, 2))

        sample_rate = audiosegment.frame_rate

        tempo_ratio = new_tempo / tempo
        print(tempo_ratio)
        y_fast = pyrb.time_stretch(y, sample_rate, tempo_ratio)
 
        channels = 2 if (y_fast.ndim == 2 and y_fast.shape[1] == 2) else 1
        y = np.int16(y_fast * 2 ** 15)

        new_seg = AudioSegment(y.tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)

        return new_seg

    def getNextMusic(self):

        mp_holistic = mp.solutions.holistic # Holistic model
        mp_drawing = mp.solutions.drawing_utils # Drawing utilities

        def mediapipe_detection(image, model):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
            image.flags.writeable = False                  # Image is no longer writeable
            results = model.process(image)                 # Make prediction
            image.flags.writeable = True                   # Image is now writeable 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
            return image, results
        def draw_styled_landmarks(image, results):
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    ) 
        def draw_landmarks(image, results):
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        def extract_keypoints(results):
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
            return np.concatenate([pose])
        
        actions = np.array(['idle', 'wave', 'jump'])

        model = Sequential() 
        model.add(GRU(256, return_sequences=True, activation='relu', input_shape=(10,132))) #10frame with 36keypoints
        model.add(GRU(128, return_sequences=True, activation='relu'))
        model.add(GRU(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        #Load Model
        model.load_weights('0504.h5')

        colors = [(245,117,16), (117,245,16), (16,117,245)]
        def prob_viz(res, actions, input_frame, colors):
            output_frame = input_frame.copy()
            blk2 = np.zeros(output_frame.shape, np.uint8)
            for num, prob in enumerate(res):
                cv2.rectangle(blk2, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
                output_frame = cv2.addWeighted(output_frame, 1.0, blk2, 0.5,1)
                cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            return output_frame

        
        threshold = 0.7

        fps = self.vid.get(cv2.CAP_PROP_FPS)
        CONSTANT_FPS_VALUE = fps * 4

        while(1):
            time.sleep(0.1)
            if(self.speedChangeFlag):
                self.speedChangeFlag = False

                self.qCrowd = []
                sequence = []
                sentence = []
                predictions = []
                score = []
                count_time = []
                flag_stop = True
                fps = self.vid.get(cv2.CAP_PROP_FPS)
                CONSTANT_FPS_VALUE = fps * 4
                with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
                    while ((self.vid.isOpened()) and flag_stop):
                        ret, image = self.vid.read()
                        image, results = mediapipe_detection(image, holistic)
                        draw_styled_landmarks(image, results)
                        count_time.append(0)
                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        sequence = sequence[-10:] 
                            
                        if len(sequence) == 10:
                            res = model.predict(np.expand_dims(sequence, axis=0))[0]
                            predictions.append(np.argmax(res))
                            score.append((np.argmax(res)+1))
                            Sum = sum(score)
                            f = Sum/len(score)
                            crowd_score = ceil(f * 100) / 100.0
                            if len(count_time)>= CONSTANT_FPS_VALUE:
                                if crowd_score >= 1.5 and crowd_score <2.25 :
                                    tempCrowd = 2
                                    if (len(self.qCrowd)<2):
                                        self.qCrowd.append(tempCrowd)
                                    else:
                                        self.qCrowd[0:2] = [self.qCrowd[1],tempCrowd]
                                    self.ui.actionLabel.setText("<html><head/><body><p><span style=\" font-size:11pt;\">Energy Constant!</span></p></body></html>")
                                    cv2.putText(image, 'play the same energy music', (0,200), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                                if crowd_score >= 2.25:    
                                    tempCrowd = 0
                                    if (len(self.qCrowd)<2):
                                        self.qCrowd.append(tempCrowd)
                                    else:
                                        self.qCrowd[0:2] = [self.qCrowd[1],tempCrowd]
                                    self.ui.actionLabel.setText("<html><head/><body><p><span style=\" font-size:11pt;\">Energy Up!</span></p></body></html>")
                                    cv2.putText(image, 'play more energetic music', (0,200), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                                if crowd_score < 1.5:
                                    tempCrowd = 1
                                    if (len(self.qCrowd)<2):
                                        self.qCrowd.append(tempCrowd)
                                    else:
                                        self.qCrowd[0:2] = [self.qCrowd[1],tempCrowd]
                                    self.ui.actionLabel.setText("<html><head/><body><p><span style=\" font-size:11pt;\">Energy Down!</span></p></body></html>")
                                    cv2.putText(image, 'play music with less energy', (0,200), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                                flag_stop = False
                            
                            if np.unique(predictions[-10:])[0]==np.argmax(res):
                                if res[np.argmax(res)] > threshold: 
                                    
                                    if len(sentence) > 0: 
                                            sentence.append(actions[np.argmax(res)])
                                    else:
                                        sentence.append(actions[np.argmax(res)])

                            if len(sentence) > 10:   
                                sentence = sentence[-10:]

                            image = prob_viz(res, actions, image, colors)

                            blk = np.zeros(image.shape, np.uint8)
                            cv2.rectangle(blk, (0,0), (640, 40), (255, 0, 0), -1)
                            image = cv2.addWeighted(image, 1.0, blk, 0.5,1)
                            cv2.putText(image, ' | '.join(sentence), (3,30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, 'Len:{} '.format(len(score)), (0,240), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(image, 'Score:{}'.format(crowd_score), (0,270), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 22, 235), 2, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('Action Recognition', image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                try:
                    hasilcrod = self.qCrowd.pop(0)
                    tempfile = self.ms.getNextMusic(crowd=hasilcrod)
                except:
                    tempfile = self.ms.getNextMusic(crowd=2)
                _,temporarySong = tempfile.rsplit('\\',1)
                chosenSong,_= temporarySong.rsplit('.',1)
                cv2.destroyAllWindows()
                for i in range(len(self.playlist)):
                    _,temporarySong = self.playlist[i][0].rsplit('/',1)
                    comparedSong,_ = temporarySong.rsplit('.',1)
                    if(comparedSong==chosenSong):
                        self.nextSongIndex = i
                        break
        
                if (self.playlist[self.nextSongIndex][1]/self.playlist[self.nowPlayingIndex][1]) != 1:     
                    self.ui.beatmatching.setText("Tempo Mismatch")

                    if(self.currentPlayer == 1):
                        self.ui.initialKey2.setText(str(self.playlist[self.nextSongIndex][8]))
                        self.ui.bpm2.setText(str(self.playlist[self.nextSongIndex][1])+" BPM") 

                    if(self.currentPlayer == 2):
                        self.ui.initialKey1.setText(str(self.playlist[self.nextSongIndex][8]))
                        self.ui.bpm1.setText(str(self.playlist[self.nextSongIndex][1])+" BPM") 

                    speedChanged = self.change_audioseg_tempo(self.playlist[self.nextSongIndex][4],self.playlist[self.nextSongIndex][1],self.playlist[self.nowPlayingIndex][1])
                    
                    self.playlist[self.nextSongIndex][3] = self.playlist[self.nextSongIndex][3] * (self.playlist[self.nextSongIndex][1]/self.playlist[self.nowPlayingIndex][1])
                    self.playlist[self.nextSongIndex][6] = self.playlist[self.nextSongIndex][6] * (self.playlist[self.nextSongIndex][1]/self.playlist[self.nowPlayingIndex][1])

                    speedChanged = speedChanged.fade_in(10000)
                    speedChanged = speedChanged.fade(to_gain=-120.0, start=int(self.playlist[self.nextSongIndex][6][len(self.playlist[self.nextSongIndex][6])-(12-1)]*1000), duration=30000)

                    self.playlist[self.nextSongIndex][5] = self.make_chunks(speedChanged,10)
                    self.playlist[self.nextSongIndex][1] = self.playlist[self.nowPlayingIndex][1] 

                if(self.currentPlayer == 1):
                    self.ui.initialKey2.setText(str(self.playlist[self.nextSongIndex][8]))
                    self.ui.bpm2.setText(str(self.playlist[self.nextSongIndex][1])+" BPM") 

                if(self.currentPlayer == 2):
                    self.ui.initialKey1.setText(str(self.playlist[self.nextSongIndex][8]))
                    self.ui.bpm1.setText(str(self.playlist[self.nextSongIndex][1])+" BPM")  
                
                self.ui.beatmatching.setText("Phase Mismatch")
                
        self.vid.release()
        cv2.destroyAllWindows()
        
    def make_chunks(self, audio_segment, chunk_length):
        """
        Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
        long. 
        if chunk_length is 50 then you'll get a list of 50 millisecond long audio
        segments back (except the last one, which can be shorter)
        """
        number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
        return [audio_segment[i * chunk_length:(i + 1) * chunk_length]
                for i in range(int(number_of_chunks))]

    def pcm2float(self, sig, dtype='float32'):
        sig = np.asarray(sig)
        if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max

    def float2pcm(self, sig, dtype='int16'):
        sig = np.asarray(sig)
        if sig.dtype.kind != 'f':
            raise TypeError("'sig' must be a float array")
        dtype = np.dtype(dtype)
        if dtype.kind not in 'iu':
            raise TypeError("'dtype' must be an integer type")

        i = np.iinfo(dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


    def equalizer(self, audiosegment, OldValue):
        OldMax = self.OldMin + 1000
        NewMin = 0
        NewMax = 70
        NewValue = (((OldValue - self.OldMin) * (NewMax - NewMin)) / (OldMax - self.OldMin)) + NewMin
        self.bquad_filter.low_shelf(audiosegment.frame_rate, self.lowCutoff, self.Q, NewValue*-1)
        samples = audiosegment.get_array_of_samples()
        newArray = self.pcm2float(samples)
        newData = newArray
        self.bquad_filter.process(newArray,newData)
        shifted_samples_array = array.array(audiosegment.array_type, self.float2pcm(newData))

        return audiosegment._spawn(shifted_samples_array)

    def callback(self, in_data, frame_count, time_info, status):

        if(self.equalizerFlag1 == 1):
            self.chunk1[self.chunkPos1] = self.equalizer(self.chunk1[self.chunkPos1],self.chunkPos1)

        if(self.equalizerFlag2 == 1):
            self.chunk2[self.chunkPos2] = self.equalizer(self.chunk2[self.chunkPos2],self.chunkPos2)

        if(self.currentPlayer == 1):
            if(self.chunkPos1 == int((self.playlist[self.nowPlayingIndex][6][len(self.playlist[self.nowPlayingIndex][6])-(12-1)]*100) - (self.playlist[self.nextSongIndex][6][0]*100))):
                self.equalizerFlag1 = 1

            if(self.chunkPos1 == int((self.playlist[self.nowPlayingIndex][6][len(self.playlist[self.nowPlayingIndex][6])-(18-1)]*100) - (self.playlist[self.nextSongIndex][6][0]*100))):
                self.OldMin = int((self.playlist[self.nowPlayingIndex][6][len(self.playlist[self.nowPlayingIndex][6])-(18-1)]*100) - (self.playlist[self.nextSongIndex][6][0]*100))
                self.ui.beatmatching.setText("Beatmatched")
                self.playSong2(songIndex=self.nextSongIndex)
            elif(self.chunkPos1 == int(self.playlist[self.nowPlayingIndex][6][len(self.playlist[self.nowPlayingIndex][6])-(34-1)]*100)):
                self.speedChangeFlag = True

        if(self.currentPlayer == 2):
            if(self.chunkPos2 == int((self.playlist[self.nowPlayingIndex][6][len(self.playlist[self.nowPlayingIndex][6])-(12-1)]*100) - (self.playlist[self.nextSongIndex][6][0]*100))):
                self.equalizerFlag2 = 1

            if(self.chunkPos2 == int((self.playlist[self.nowPlayingIndex][6][len(self.playlist[self.nowPlayingIndex][6])-(18-1)]*100) - (self.playlist[self.nextSongIndex][6][0]*100))):
                self.OldMin = int((self.playlist[self.nowPlayingIndex][6][len(self.playlist[self.nowPlayingIndex][6])-(18-1)]*100) - (self.playlist[self.nextSongIndex][6][0]*100))
                self.ui.beatmatching.setText("Beatmatched")
                self.playSong1(songIndex=self.nextSongIndex)
            elif(self.chunkPos2 == int(self.playlist[self.nowPlayingIndex][6][len(self.playlist[self.nowPlayingIndex][6])-(34-1)]*100)):
                self.speedChangeFlag = True
        
        #To avoid EoF
        if(not self.pauseState1 and self.chunkPos1 >= len(self.chunk1)-2):
            self.stopSong1()
        if(not self.pauseState2 and self.chunkPos2 >= len(self.chunk2)-2):
            self.stopSong2()

        if(not self.pauseState1 and not self.pauseState2):
            dataTemp = self.chunk1[self.chunkPos1].overlay(self.chunk2[self.chunkPos2])
            data = dataTemp._data
            self.chunkPos1 += 1 
            self.chunkPos2 += 1
        elif(not self.pauseState1 and self.pauseState2):
            self.chunk1[self.chunkPos1] += self.volume1
            data = self.chunk1[self.chunkPos1]._data
            self.chunkPos1 += 1
        elif(self.pauseState1 and not self.pauseState2):
            self.chunk2[self.chunkPos2] += self.volume2
            data = self.chunk2[self.chunkPos2]._data
            self.chunkPos2 += 1
        else:
            try:
                data = bytes(len(self.chunk1[self.chunkPos1]._data))
            except:
                data = bytes(len(self.chunk2[self.chunkPos2]._data))

        self.curMin1,self.curSec1 = divmod(self.chunkPos1/100, 60)
        self.currentTime1 = '{:02d}:{:02d}'.format(int(self.curMin1),int(self.curSec1))
        self.ui.songLength1.setText(self.currentTime1)
        self.curMin2,self.curSec2 = divmod(self.chunkPos2/100, 60)
        self.currentTime2 = '{:02d}:{:02d}'.format(int(self.curMin2),int(self.curSec2))
        self.ui.songLength2.setText(self.currentTime2)
        if(not self.sliderModified and self.chunkPos1%100 == 0 ):
            self.ui.uselessButton.click()
            self.ui.songLengthSlider1.setValue(self.chunkPos1)
        if(not self.sliderModified and self.chunkPos2%100 == 0):
            self.ui.uselessButton.click()
            self.ui.songLengthSlider2.setValue(self.chunkPos2)
        return(data, pyaudio.paContinue)

    def getSelectedIndex1(self):
        return [x.row() for x in self.ui.songList.selectedIndexes()][0]
    
    def getSelectedIndex2(self):
        return [x.row() for x in self.ui.songList2.selectedIndexes()][0]

    def songSlider1Pressed(self):
        self.sliderModified = True
        if(self.playedAlready1):
            self.pauseSong1()
        self.ui.songLengthSlider1.sliderReleased.connect(self.setSongSlider1)

    def songSlider2Pressed(self):
        self.sliderModified = True
        if(self.playedAlready2):
            self.pauseSong2()
        self.ui.songLengthSlider2.sliderReleased.connect(self.setSongSlider2)
    
    def setSongSlider1(self):
        self.chunkPos1 = self.ui.songLengthSlider1.value()
        if(self.playedAlready1):
            self.unpauseSong1()
        else:
            self.playSong1() 
        self.sliderModified = False

    def setSongSlider2(self):
        self.chunkPos2 = self.ui.songLengthSlider2.value()
        if(self.playedAlready2):
            self.unpauseSong2()
        else:
            self.playSong2() 
        self.sliderModified = False

    def songDetails1(self,index=None):
        if (index is None):
            self.playNow1 = self.playlist[self.getSelectedIndex1()][0]
        else:
            self.playNow1 = self.playlist[index][0]
            self.ui.songList.setCurrentRow(index)

        self.oriBpm1 = self.playlist[self.getSelectedIndex1()][1]

        self.songIndex1 = self.getSelectedIndex1() 
        self.nowPlayingIndex = self.songIndex1
        
        self.sound1 = self.playlist[self.getSelectedIndex1()][4]
        self.chunk1 = self.playlist[self.getSelectedIndex1()][5]

        self.ui.actionLabel.setText("<html><head/><body><p><span style=\" font-size:11pt;\">Action Recognition</span></p></body></html>")
        self.ui.energyCounter.setText(str(self.playlist[self.getSelectedIndex1()][7]))
        self.ui.initialKey1.setText(str(self.playlist[self.getSelectedIndex1()][8]))

        tempName1 = QUrl.fromLocalFile(self.playNow1)
        self.songName1,self.songFormat1= tempName1.fileName().rsplit('.',1)
        self.ui.songPlayed1.setText(self.songName1)
        self.ui.bpm1.setText(str(self.playlist[self.getSelectedIndex1()][1])+" BPM")

        self.ui.songLengthSlider1.setRange(0,len(self.chunk1)-1)

    def songDetails2(self,index=None):
        if (index is None):
            self.playNow2 = self.playlist[self.getSelectedIndex2()][0]
        else:
            self.playNow2 = self.playlist[index][0]
            self.ui.songList2.setCurrentRow(index)

        self.oriBpm2 = self.playlist[self.getSelectedIndex2()][1]

        self.songIndex2 = self.getSelectedIndex2() 
        self.nowPlayingIndex = self.songIndex2

        self.sound2 = self.playlist[self.getSelectedIndex2()][4]
        self.chunk2 = self.playlist[self.getSelectedIndex2()][5]

        self.ui.actionLabel.setText("<html><head/><body><p><span style=\" font-size:11pt;\">Action Recognition</span></p></body></html>")
        self.ui.energyCounter.setText(str(self.playlist[self.getSelectedIndex2()][7]))
        self.ui.initialKey2.setText(str(self.playlist[self.getSelectedIndex2()][8]))

        tempName2 = QUrl.fromLocalFile(self.playNow2)
        self.songName2,self.songFormat1= tempName2.fileName().rsplit('.',1)
        self.ui.songPlayed2.setText(self.songName2)
        self.ui.bpm2.setText(str(self.playlist[self.getSelectedIndex2()][1])+" BPM")

        self.ui.songLengthSlider2.setRange(0,len(self.chunk2)-1)

    def addToPlaylist(self,f):
        self.totalSong = len(f[0])
        for i in range(len(f[0])):
            try:
                self.cache = pd.read_csv('storage.csv', encoding='ISO-8859-1')
                print("Storage Found !")
            except:
                df = pd.DataFrame(columns=['songname','bpm','beatpos','downbeat']).set_index('songname')
                df.to_csv('storage.csv')
                self.cache = pd.read_csv('storage.csv', encoding='ISO-8859-1')
                print("Storage Is Not Found !")

            tempSongname = QUrl.fromLocalFile(f[0][i])
            tempSongname,_ = tempSongname.fileName().rsplit('.',1)
            self.ms.addSong(f[0][i])
            energy = self.ms.real_mstorage.getFeature(['energy'], os.path.normpath(f[0][i]))
            initialKey = self.ms.real_mstorage.getFeature(['key'], os.path.normpath(f[0][i]))
            CONSTANT_KEY = ['1A','2A','3A','4A','5A','6A','7A','8A','9A','10A','11A','12A','1B','2B','3B','4B','5B','6B','7B','8B','9B','10B','11B','12B']
            print(energy['energy'])
            print(CONSTANT_KEY[initialKey['key']])
            
            #Convert To WAV
            print(f[0][i])
            print(tempSongname)
            try:
                subprocess.run(['ffmpeg', '-i', f[0][i].encode('utf-8') , tempSongname + '.wav'], capture_output = True, text=True, input="y")        
            except:
                pass               

            #Save song to songlist
            self.ui.songList.insertItem(self.index, tempSongname)
            self.ui.songList.setCurrentRow(0)
            self.ui.songList2.insertItem(self.index, tempSongname)
            self.ui.songList2.setCurrentRow(0)

            if(self.cache['songname']==tempSongname).any():
                print("Song Annotation Cache Found !")
                foundSong = self.cache.loc[self.cache['songname']==tempSongname]
                tempo = foundSong['bpm'].item()
                beatTimes = numpy.array(ast.literal_eval(foundSong['beatpos'].item()))
                cleanedDownbeat = numpy.array(ast.literal_eval(foundSong['downbeat'].item()))
            else:
                print("Song Is Not Annotated !")
                #BeatTracking and BPM
                act = madmom.features.beats.RNNBeatProcessor()(f[0][i])
                proc = madmom.features.beats.BeatTrackingProcessor(fps=100)(act)
                m_res = scipy.stats.linregress(np.arange(len(proc)),proc)
                beat_step = m_res.slope
                tempo = round(60/beat_step)
                beatTimes = proc

                #Downbeat Tracking  
                procDown = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[4, 4], fps=100)
                actDown = madmom.features.downbeats.RNNDownBeatProcessor()(f[0][i]) 
                downbeat = procDown(actDown) 
                downbeat = downbeat[downbeat[:,1]==1]
                cleanedDownbeat = downbeat[:,0]

                #Save annotated music
                data = [tempSongname,tempo,beatTimes,cleanedDownbeat]
                df = pd.DataFrame([data],columns=['songname','bpm','beatpos','downbeat']).set_index('songname')
                df['beatpos']=df['beatpos'].map(list)
                df['downbeat']=df['downbeat'].map(list)

                if os.path.isfile('./storage.csv'):
                    df.to_csv('storage.csv', mode='a', encoding='ISO-8859-1', header=False)
                else:
                    df.to_csv('storage.csv',encoding='ISO-8859-1')                    

            soundOri = AudioSegment.from_wav(tempSongname + '.wav')
            soundOri = soundOri.fade_in(10000)
            soundOri = soundOri.fade(to_gain=-120.0, start=int(cleanedDownbeat[len(cleanedDownbeat)-(12-1)]*1000), duration=30000)
            chunk = self.make_chunks(soundOri,10)

            #Store information gathered into playlist
            self.playlist[self.index].append(f[0][i])
            self.playlist[self.index].append(tempo)
            self.playlist[self.index].append([])
            self.playlist[self.index].append(beatTimes)
            self.playlist[self.index].append(soundOri)
            self.playlist[self.index].append(chunk)
            self.playlist[self.index].append(cleanedDownbeat)
            self.playlist[self.index].append(energy.get('energy')+1)
            self.playlist[self.index].append(CONSTANT_KEY[initialKey['key']])
            self.playlist.append([])
            
            self.index +=1
            print(self.index)
        
    def addFile(self):
        filter = "MP3 (*.mp3);;WAV (*.wav)"
        self.file = QtWidgets.QFileDialog()
        self.file.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        self.filename = QtWidgets.QFileDialog.getOpenFileNames(filter=filter)
        self.addToPlaylist(self.filename)       

    def playSong1(self,songIndex=None):
        if(self.pauseState1):
            if(not self.playedAlready1):
                try:
                    self.currentPlayer = 1
                    self.equalizerFlag1 = 0
                    if not songIndex:
                        self.songDetails1()
                        self.ms.manualNextMusic(self.playlist[self.getSelectedIndex1()][0])
                    else:
                        self.songDetails1(songIndex)
                        self.ms.manualNextMusic(self.playlist[songIndex][0])

                    if(not self.pyaudioInitiated):
                        self.thread1.start()
                        p = pyaudio.PyAudio()
                        self.stream = p.open(format=p.get_format_from_width(self.sound1.sample_width),
                                channels=self.sound1.channels,
                                frames_per_buffer=ceil(len(self.chunk1[self.chunkPos1]._data)/4),
                                rate=self.sound1.frame_rate,
                                output=True,
                                stream_callback=self.callback)
                        self.pyaudioInitiated = True
                    self.playedAlready1 = True
                    self.pauseState1 = False
                    self.ui.play1.setIcon(QtGui.QIcon("pause.png"))
                except:
                    error_dialog = QtWidgets.QErrorMessage()
                    error_dialog.showMessage('Choose A Song First')
                    error_dialog.exec_()
            else:   
                self.unpauseSong1()
        else:
            self.pauseSong1()

    def playSong2(self,songIndex=None):
        if(self.pauseState2):
            if(not self.playedAlready2):
                try:
                    self.currentPlayer = 2
                    self.equalizerFlag2 = 0
                    if not songIndex:
                        self.songDetails2()
                        self.ms.manualNextMusic(self.playlist[self.getSelectedIndex2()][0])
                    else:
                        self.songDetails2(songIndex)
                        self.ms.manualNextMusic(self.playlist[songIndex][0])
                    
                    if(not self.pyaudioInitiated):                   
                        self.thread1.start()
                        p = pyaudio.PyAudio()
                        self.stream = p.open(format=p.get_format_from_width(self.sound2.sample_width),
                                channels=self.sound2.channels,
                                frames_per_buffer=ceil(len(self.chunk2[self.chunkPos2]._data)/4),
                                rate=self.sound2.frame_rate,
                                output=True,
                                stream_callback=self.callback)
                        self.pyaudioInitiated = True
                    self.playedAlready2 = True
                    self.pauseState2 = False
                    self.ui.play2.setIcon(QtGui.QIcon("pause.png"))
                except:
                    error_dialog = QtWidgets.QErrorMessage()
                    error_dialog.showMessage('Choose A Song First')
                    error_dialog.exec_()
            else:   
                self.unpauseSong2()
        else:
            self.pauseSong2()

    def pauseSong1(self):
        self.pauseState1 = True
        self.ui.play1.setIcon(QtGui.QIcon("play.png"))

    def pauseSong2(self):
        self.pauseState2 = True
        self.ui.play2.setIcon(QtGui.QIcon("play.png"))

    def unpauseSong1(self):
        self.pauseState1 = False
        self.ui.play1.setIcon(QtGui.QIcon("pause.png"))

    def unpauseSong2(self):
        self.pauseState2 = False
        self.ui.play2.setIcon(QtGui.QIcon("pause.png"))
    
    def stopSong1(self): 
        self.equalizerFlag1 = 0  
        self.playedAlready1 = False   
        self.pauseState1 = True
        self.chunkPos1 = 0
        self.length1 = '00:00'
        self.ui.songLengthSlider1.setValue(0)
        self.ui.songLength1.setText(self.length1)  
        self.ui.play1.setIcon(QtGui.QIcon("play.png")) 
        self.ui.bpm1.setText("")
        self.ui.initialKey1.setText("")
        self.ui.beatmatching.setText("")

    def stopSong2(self):
        self.equalizerFlag2 = 0  
        self.playedAlready2 = False   
        self.pauseState2 = True
        self.chunkPos2 = 0
        self.length2 = '00:00'
        self.ui.songLengthSlider2.setValue(0)
        self.ui.songLength2.setText(self.length2)  
        self.ui.play2.setIcon(QtGui.QIcon("play.png"))
        self.ui.bpm2.setText("")
        self.ui.initialKey2.setText("")
        self.ui.beatmatching.setText("")

    def setVolume1(self):
        self.volume1 = (self.ui.volume1.value())

    def setVolume2(self):
        self.volume2 = (self.ui.volume2.value())

    def rewindSong1(self):
        if(self.time1 < 5 and (self.getSelectedIndex1()) != 0):
            if(self.songIndex1-1 <= 0):
                self.ui.songList.setCurrentRow(0)
            else:
                self.ui.songList.setCurrentRow(self.songIndex1-1)
            self.stopSong1()
            self.playSong1()
        else:
            self.stopSong1()
            self.playSong1()

    def rewindSong2(self):
        if(self.time2 < 5 and (self.getSelectedIndex2()) != 0):
            if(self.songIndex2-1 <= 0):
                self.ui.songList2.setCurrentRow(0)
            else:
                self.ui.songList2.setCurrentRow(self.songIndex2-1)
            self.stopSong2()
            self.playSong2()
        else:
            self.stopSong2()
            self.playSong2()

    def nextSong1(self,songIndex=None):
        if(self.songIndex1+1 >= self.index):
            self.ui.songList.setCurrentRow(self.index-1)
        else:
            self.ui.songList.setCurrentRow(self.songIndex1+1)
        self.stopSong1()
        self.playSong1()  

    def nextSong2(self,songIndex=None):
        if(self.songIndex2+1 >= self.index):
            self.ui.songList2.setCurrentRow(self.index-1)
        else:
            self.ui.songList2.setCurrentRow(self.songIndex2+1)
        self.stopSong2()
        self.playSong2()  
 
if __name__ == "__main__":
        play = MusicPlayer()
        
        
        