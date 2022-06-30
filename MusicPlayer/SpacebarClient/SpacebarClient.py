from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QUrl,QThread
import SpacebarCounterGUI as SpacebarCounter
import sys
import pyrebase 

angka_jumlah = 0
ptr_cntr = 0
petunjuk = 0
petunjuk2 = 0
config = {
  "apiKey": "AAAAbS7bktA:APA91bFFYNC2jlZZmsE9jV0lLAuRmilHQiEG7PKyLl0jx4SZ6w-IbpBg0CtI_XNpWW8WkAcvjm4QKsW7koVPJkIzd3WRSQrnorTdi0Yjs_kAEwicHvqgRSK-aOiSuhse4FNlGUxGjD_j",
  "authDomain": "aidjtest.firebaseapp.com",
  "databaseURL": "https://aidjtest-default-rtdb.asia-southeast1.firebasedatabase.app/",
  "storageBucket": "aidjtest.appspot.com"
}
firebase = pyrebase.initialize_app(config)
db = firebase.database()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.queueKey = []

    def keyPressEvent(self, event):
        pressed_button = event.key()
        if pressed_button == QtCore.Qt.Key_Escape:
            self.close()
        elif pressed_button == QtCore.Qt.Key_Space:
            global angka_jumlah
            global ptr_cntr
            angka_jumlah += 1
            try:
                ptr_cntr.setText(str(angka_jumlah))
                db.child("sb-ctr").child("value").set(int(angka_jumlah))
            except:
                angka_jumlah -= 1
                print("Not Now!")

class SpacebarClient():
    def __init__(self):
        global petunjuk
        global petunjuk2
        #Build User Interface Using PyQT5 
        self.app = QtWidgets.QApplication(sys.argv)
        self.SplashScreen = MainWindow()
        self.ui = SpacebarCounter.Ui_SplashScreen()

        self.ui.setupUi(self.SplashScreen)
        self.SplashScreen.setWindowFlags(QtCore.Qt.FramelessWindowHint) # Remove title bar
        self.SplashScreen.setAttribute(QtCore.Qt.WA_TranslucentBackground) # Set background to transparent
        self.SplashScreen.show()
        petunjuk = self.ui.spacebarCounter
        petunjuk2 = self.ui.spacebarLabel
        sys.exit(self.app.exec_())

                
        
def stream_handler(message):
    global angka_jumlah
    global ptr_cntr
    global petunjuk
    print(message["event"]) # put
    print(message["path"]) # /-K7yGTTEp7O549EzTYtI
    print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
    try:
        crt_flag = -1
        crt_value = 0
        if (message['path']=='/'): #00
            crt_flag = message['data']['flag']
            crt_value = message['data']['value']
        elif (message['path']=='/value'): #01
            crt_value = message['data']
        elif (message['path']=='/flag'): #02
            crt_flag = message['data']
        if (crt_flag==True):
            ptr_cntr = petunjuk
            petunjuk2.setText("<html><head/><body><p><span style=\" font-size:11pt;\">Press Spacebar!</span></p></body></html>")
        elif(crt_flag==False):
            ptr_cntr = 0
            petunjuk2.setText("<html><head/><body><p><span style=\" font-size:11pt;\">Get Ready To Press Spacebar</span></p></body></html>")
        if (ptr_cntr!=0):
            angka_jumlah = crt_value
        petunjuk.setText(str(crt_value))
    except Exception as e:
        print(e)

if __name__ == "__main__":
    
    my_stream = db.child("sb-ctr").stream(stream_handler)
    # print(db.child("sb-ctr").child('value').set(0))
    # print(db.child("sb-ctr").child('flag').set(False))
    play = SpacebarClient()
    
    my_stream.close()
        
        