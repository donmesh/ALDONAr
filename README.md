# ALDONAr
Code for ALDONAr: A Hybrid Solution for Sentence-Level Aspect Based Sentiment Analysis using a Lexicalized Domain Ontology and a Regularized Neural Attention Model

All software is written in Python 3.6 (https://www.python.org/) and makes use of the TensorFlow 1.15.0 framework (https://www.tensorflow.org/).

## Installation Instructions:
### Download required files:
1. Download ontology: https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData
2. Download SemEval2016 Datasets: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
3. Download SemEval2015 Datasets: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
4. Download GloVe word embeddings: http://nlp.stanford.edu/data/glove.42B.300d.zip

### Setup Environment

1. Download and install Python 3.6

2. Install the required packages from the requirements.txt file by running the following command in a command line: pip install -r requirements.txt

3. Place "ontology.owl" in "./Ontology/" folder

4. Place data files in "./Data/" folder.

5. For all benchmark models place GloVe word embeddings in ".Embeddings/" folder.

### Run Software
If necessary any of FLAGS can be changed inside each model, otherwise they can be run in a terminal.

## Software explanation:
- DataReader.py: contains a Reader designed to extract data from the xml files, as well as containing several plotting functions
- BaseA.py: the main file to run the BaseA classification
- BaseB.py: the main file to run the BaseB classification
- BaseC.py: the main file to run the BaseC classification
- CABASC.py: the main file to run the CABASC classification
- CTX-LSTM.py: the main file to run the CTX-LSTM classification
- CTX-BGRU.py: the main file to run the CTX-BGRU classification
- CTX-BLSTM.py: the main file to run the CTX-BLSTM classification
- DBGRU.py: the main file to run the DBGRU classification
- ALDONA.py: the main file to run the ALDONA classification
- ALDONAr_base.py: the main file to run the ALDONAr_base classification
- ALDONAr.py: the main file to run the ALDONAr classification
