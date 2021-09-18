# backend

# Installation
Install `eng.traineddata` from [url](https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true) to the path `/usr/share/tesseract-ocr/4.00/tessdata`


```
pip install git+https://github.com/ramsrigouthamg/Questgen.ai
pip install sense2vec==1.0.2
pip install git+https://github.com/boudinfl/pke.git

python -m nltk.downloader universal_tagset
python -m spacy download en

wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
tar -xvf  s2v_reddit_2015_md.tar.gz
ls s2v_old
```