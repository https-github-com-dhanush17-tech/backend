# backend

Author: [Ani Aggarwal](www.github.com/AniAggarwal)

# Installation
```
sudo apt update
sudo apt upgrade
sudo apt -y install ffmpeg libs6 libxext6
```

Install `eng.traineddata` from [url](https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true) to the path `/usr/share/tesseract-ocr/4.00/tessdata`
```
sudo apt install tesseract-ocr
wget https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true eng.traineddata
mv eng.traineddata /usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata
```


```
pip3 install git+https://github.com/ramsrigouthamg/Questgen.ai
pip3 install sense2vec==1.0.3
pip3 install git+https://github.com/boudinfl/pke.git

python3 -m nltk.downloader universal_tagset
python3 -m spacy download en

wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
tar -xvf  s2v_reddit_2015_md.tar.gz
ls s2v_old
```

```
pip3 install flask
pip3 install Pillow
pip3 install pytesseract
pip3 install autocorrect
pip3 install SpeechRegonition
```