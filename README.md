AI DJ System For Electronic Dance Music
=======================================

AI DJ system is an automated DJ system combining action recognition, song selection, and beatmatching with equalizer mixing



Prerequisites
-------------

AI DJ system was tested on Python 3.8.5 and the following packages have to be installed:

- `madmom==0.16.1`
- `librosa==0.8.1`
- `mediapipe==0.8.8.1`
- `opencv_python==4.5.4.58`
- `pandas==1.3.4`
- `pydub==0.25.1`
- `PyQt5==5.15.7`
- `pyrubberband==0.3.0`
- `scikit_learn==1.1.1`
- `scipy==1.7.1`
- `yodel==0.3.0`
- `tensorflow==2.6.0`

AI DJ system also require NVIDIA GPU for optimal performance.

Installation
-------------
Open the command prompt, install git LFS, and clone the repository
```
git lfs install
git clone https://github.com/akriptonn/aidjsystem.git
```
Create an environment for AIDJ-System
```
conda create -n AIDJ-System python=3.8.5 -y
conda activate AIDJ-System
conda install -c anaconda tensorflow-gpu=2.6.0
```
Installation steps as below:
```
cd aidjsystem/MusicPlayer
pip install pipwin
pipwin install pyaudio
pip install -r requirements.txt
```
and run the code by:
```
python main.py
```

How to use
-----------
- First run the main program
```
python main.py
```
- Wait until the loading process has been done, press add file, and then choose the songs that you want 
- We provide 100 edm songs example, please contact us if you need: puffcornmusic@gmail.com

![](https://i.imgur.com/A4EsreY.png)
- Choose the song and press play button

![](https://i.imgur.com/vDuMHVI.png)
- Press ESC button to close the program
- Enjoy AI DJ :D 
