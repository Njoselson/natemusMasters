import numpy as np
import pandas as pd
np.random.seed(7)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Input, Concatenate, Dropout, Bidirectional, Reshape, Flatten
from keras import optimizers
from keras.models import load_model, Model
from keras import callbacks
from matplotlib import pyplot
import emoji
import json, argparse, os
import re
import io
import sys
sys.path.append(os.getcwd())
from helper_functions import *
from bidir_model import *

#get all test data
print("Processing test/train data for first model...")
trainIndices, trainTexts, labels, u1_train, u2_train, u3_train, smil_train = preprocessData(trainDataPath, mode="train")
testIndices, testTexts, testLabels, u1_test, u2_test, u3_test, smil_test = preprocessData(testDataPath, mode="train")

print("Extracting tokens...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(u1_train+u2_train+u3_train)

print("Right format...")
u1_testSequences, u2_testSequences, u3_testSequences, smil_testSeq = tokenizer.texts_to_sequences(u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test), tokenizer.texts_to_sequences(smil_test)
u1_testData = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
u2_testData = pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
u3_testData = pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
smil_testData = pad_sequences(smil_testSeq, maxlen=20)
testLabels = to_categorical(np.asarray(testLabels))

print("Preprocess for second model...")
trainIndices, labels2, t1, t2, t3, len1, len2, len3, case1, case2, case3, smil1, smil2, smil3 = preprocessData2(trainDataPath)
ind_test, labels2_test, t1_test, t2_test, t3_test, len1_test, len2_test, len3_test, case1_test, case2_test, case3_test, smil1_test, smil2_test, smil3_test = preprocessData2(testDataPath)

print("Extracting tokens for second model...")
tokenizerModelTwo = Tokenizer(num_words=MAX_NB_WORDS)
tokenizerModelTwo.fit_on_texts(t1+t2+t3)
wordIndexModelTwo = tokenizerModelTwo.word_index

t1, t2, t3 = tokenizerModelTwo.texts_to_sequences(t1), tokenizerModelTwo.texts_to_sequences(t2), tokenizerModelTwo.texts_to_sequences(t3)
t1_test, t2_test, t3_test = tokenizerModelTwo.texts_to_sequences(t1_test), tokenizerModelTwo.texts_to_sequences(t2_test), tokenizerModelTwo.texts_to_sequences(t3_test)

t1, t2, t3 = pad_sequences(t1, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t2, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t3, maxlen=MAX_SEQUENCE_LENGTH)
t1_test, t2_test, t3_test = pad_sequences(t1_test, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t2_test, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t3_test, maxlen=MAX_SEQUENCE_LENGTH)

meta_data = np.asarray([len1, len2, len3, case1, case2, case3, smil1, smil2, smil3]).T
meta_data_test = np.asarray([len1_test, len2_test, len3_test, case1_test, case2_test, case3_test, smil1_test, smil2_test, smil3_test]).T

#Load models
print("Load models...")
model1 = load_model('EP100_LR100e-5_LDim128_BS2500.h5')
model2 = load_model('../b_smileyZeroEmbeddings_F1_73/EP2_LR100e-5_LDim128_BS200.h5')

print("Make predictions...")
preds1 = model1.predict([t1_test, t2_test, t3_test, meta_data_test], batch_size=BATCH_SIZE)
preds2 = model2.predict([u1_testData, u2_testData, u3_testData, smil_testData], batch_size=BATCH_SIZE)
df = pd.DataFrame({'Pred model 1': preds1, 'Pred model 2': preds2, 'Correct': testLabels})
df.to_csv('predictions.csv', sep='\t')
