# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SpacebarCounter.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SplashScreen(object):
    def setupUi(self, SplashScreen):
        SplashScreen.setObjectName("SplashScreen")
        SplashScreen.resize(339, 362)
        SplashScreen.setMouseTracking(True)
        SplashScreen.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(SplashScreen)
        self.centralwidget.setObjectName("centralwidget")
        self.circularProgressBarBase = QtWidgets.QFrame(self.centralwidget)
        self.circularProgressBarBase.setGeometry(QtCore.QRect(10, 10, 321, 331))
        self.circularProgressBarBase.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.circularProgressBarBase.setFrameShadow(QtWidgets.QFrame.Raised)
        self.circularProgressBarBase.setObjectName("circularProgressBarBase")
        self.circularProgress = QtWidgets.QFrame(self.circularProgressBarBase)
        self.circularProgress.setGeometry(QtCore.QRect(10, 10, 301, 300))
        self.circularProgress.setStyleSheet("QFrame{\n"
"    border-radius:150px;\n"
"    background-color: qconicalgradient(cx:0.5, cy:0.5, angle:90, stop:0.749 rgba(85, 170, 255, 0), stop:0.75 rgba(85, 170, 255, 255));\n"
"\n"
"}")
        self.circularProgress.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.circularProgress.setFrameShadow(QtWidgets.QFrame.Raised)
        self.circularProgress.setObjectName("circularProgress")
        self.container = QtWidgets.QFrame(self.circularProgress)
        self.container.setGeometry(QtCore.QRect(10, 10, 281, 291))
        self.container.setStyleSheet("QFrame{\n"
"border-radius:140px;\n"
"background-color: rgb(85, 85, 127);\n"
"}")
        self.container.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.container.setFrameShadow(QtWidgets.QFrame.Raised)
        self.container.setObjectName("container")
        self.title = QtWidgets.QLabel(self.container)
        self.title.setGeometry(QtCore.QRect(80, 30, 120, 21))
        font = QtGui.QFont()
        font.setFamily("Raleway Medium")
        self.title.setFont(font)
        self.title.setStyleSheet("background-color:none;\n"
"color:#FFFFFF;\n"
"")
        self.title.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.title.setObjectName("title")
        self.circularBg_2 = QtWidgets.QFrame(self.container)
        self.circularBg_2.setGeometry(QtCore.QRect(20, 100, 241, 111))
        font = QtGui.QFont()
        font.setKerning(True)
        self.circularBg_2.setFont(font)
        self.circularBg_2.setStyleSheet("QFrame{\n"
"border-radius : 195px;\n"
"background-color: rgba(85, 85, 127, 120);\n"
"}")
        self.circularBg_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.circularBg_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.circularBg_2.setObjectName("circularBg_2")
        self.spacebarCounter = QtWidgets.QLabel(self.circularBg_2)
        self.spacebarCounter.setGeometry(QtCore.QRect(60, 40, 121, 61))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(38)
        font.setBold(False)
        font.setWeight(50)
        self.spacebarCounter.setFont(font)
        self.spacebarCounter.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.spacebarCounter.setAlignment(QtCore.Qt.AlignCenter)
        self.spacebarCounter.setObjectName("spacebarCounter")
        self.spacebarCounter.setStyleSheet("color:#ffffff;\n"
"")
        self.spacebarLabel = QtWidgets.QLabel(self.circularBg_2)
        self.spacebarLabel.setGeometry(QtCore.QRect(10, 10, 221, 21))
        font = QtGui.QFont()
        font.setFamily("Roboto Light")
        self.spacebarLabel.setFont(font)
        self.spacebarLabel.setStyleSheet("background-color:none;\n"
"color:#ffffff;\n"
"")
        self.spacebarLabel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.spacebarLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.spacebarLabel.setObjectName("spacebarLabel")
        self.circularBg = QtWidgets.QFrame(self.circularProgressBarBase)
        self.circularBg.setGeometry(QtCore.QRect(10, 10, 301, 311))
        self.circularBg.setStyleSheet("QFrame{\n"
"border-radius : 150px;\n"
"background-color: rgba(85, 85, 127, 120);\n"
"}")
        self.circularBg.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.circularBg.setFrameShadow(QtWidgets.QFrame.Raised)
        self.circularBg.setObjectName("circularBg")
        self.circularBg.raise_()
        self.circularProgress.raise_()
        SplashScreen.setCentralWidget(self.centralwidget)

        self.retranslateUi(SplashScreen)
        QtCore.QMetaObject.connectSlotsByName(SplashScreen)

    def retranslateUi(self, SplashScreen):
        _translate = QtCore.QCoreApplication.translate
        SplashScreen.setWindowTitle(_translate("SplashScreen", "MainWindow"))
        self.title.setText(_translate("SplashScreen", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600; color:#ababff;\">AI </span><span style=\" font-size:10pt;\">DJ</span></p><p><br/></p></body></html>"))
        self.spacebarCounter.setText(_translate("SplashScreen", "0"))
        self.spacebarLabel.setText(_translate("SplashScreen", "<html><head/><body><p><span style=\" font-size:11pt;\">Get Ready To Press Spacebar</span></p></body></html>"))


