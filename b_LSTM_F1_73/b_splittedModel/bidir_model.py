# %% # Smiley Separation Model
#' This is an improvement to our smiley separation
#' which now takes the 3 turns as separate inputs
#' into the LSTM layers.


#+ setup, echo=False
import numpy as np
np.random.seed(7)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Input, Concatenate, Dropout, Bidirectional, Reshape, Flatten
from keras import optimizers
from keras.models import load_model, Model
from matplotlib import pyplot
import pandas as pd
import emoji
import json, argparse, os
import re
import io
import sys
sys.path.append(os.getcwd())
from helper_functions import *



global trainDataPath, testDataPath, solutionPath, gloveDir
global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE, EARLY_STOPPING

#parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
#parser.add_argument('-config', help='Config to read details', required=True)
#args = parser.parse_args()

with open('testBaseline.config') as configfile:
    config = json.load(configfile)

trainDataPath = config["train_data_path"]
validationDataPath = config["validation_data_path"]
testDataPath = config["test_data_path"]
solutionPath = config["solution_path"]
gloveDir = config["glove_dir"]

trainOther = config["binary_train"]
testOther = config["binary_test"]
valOther = config["binary_val"]

NUM_FOLDS = config["num_folds"]
NUM_CLASSES = config["num_classes"]
MAX_NB_WORDS = config["max_nb_words"]
MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
EMBEDDING_DIM = config["embedding_dim"]
BATCH_SIZE = config["batch_size"]
LSTM_DIM = config["lstm_dim"]
DROPOUT = config["dropout"]
LEARNING_RATE = config["learning_rate"]
NUM_EPOCHS = config["num_epochs"]
EARLY_STOPPING = config["early_stopping"]
label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}

#' ## Config
#' This model was produced with the following configs:

#+ configs, echo=False
print('Embedding Dim = %d' %EMBEDDING_DIM)
print('Batch Size = %d' %BATCH_SIZE)
print('LSTM Dim = %d' %LSTM_DIM)
print('Dropout = %d' %DROPOUT)
print('Learning Rate = %d' %LEARNING_RATE)
print('Num Epochs = %d' %NUM_EPOCHS)
print('Early Stopping = ' + EARLY_STOPPING)



#+ defs, echo=False

def preprocessData(dir):
    df = pd.read_csv(dir, sep='\t')

    df.label = df.label.apply(other_or_emo)
    labels = df.label.tolist()

    df['t1_len'] = df['turn1'].apply(count_length)
    df['t2_len'] = df['turn2'].apply(count_length)
    df['t3_len'] = df['turn3'].apply(count_length)

    df['t1_upper'] = df['turn1'].apply(count_upper_case)
    df['t2_upper'] = df['turn2'].apply(count_upper_case)
    df['t3_upper'] = df['turn3'].apply(count_upper_case)

    df['turn1'] = df['turn1'].apply(str2emoji).apply(remove_emoji).apply(lambda x: x.lower()).apply(preprocessString)
    df['turn2'] = df['turn2'].apply(str2emoji).apply(remove_emoji).apply(lambda x: x.lower()).apply(preprocessString)
    df['turn3'] = df['turn3'].apply(str2emoji).apply(remove_emoji).apply(lambda x: x.lower()).apply(preprocessString)

    ind = df.id.tolist()

    t1, t2, t3 = df.turn1.tolist(), df.turn2.tolist(), df.turn3.tolist()
    len1, len2, len3 = df.t1_len.tolist(), df.t3_len.tolist(), df.t3_len.tolist()
    case1, case2, case3 = df.t1_upper.tolist(), df.t2_upper.tolist(), df.t3_upper.tolist()
    return ind, labels, t1, t2, t3, len1, len2, len3, case1, case2, case3


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)

    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)

    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1

def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 200 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(gloveDir, 'glove.twitter.27B.200d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1.
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
        else:
            embeddingMatrix[i] = np.random.random(EMBEDDING_DIM)

    return embeddingMatrix

######### Smily embeddings ##################
def getSmileyEmbeddings(wordIndex):

    embeddingsIndex = {}

    with io.open(os.path.join(gloveDir, 'emoji2vec.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    smileyEmbeddings = np.zeros((len(wordIndex)+1, 300))
    for smiley, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(smiley)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            smileyEmbeddings[i] = embeddingVector
        else:
            smileyEmbeddings[i] = np.zeros(300)
            #smileyEmbeddings[i] = np.random.random(300)

    return smileyEmbeddings
#############################################

#' # Model Specification
#' # An extra hidden layer which is supposed to extract the important information from the
#' # first embedding then add information from smilys

#+ model, echo=True
def buildModel(embeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : three layer lstm model with one layer of meta data
    """

    t1 = Input(shape=(100,), dtype='int32', name='conv_turn_1')
    t2 = Input(shape=(100,), dtype='int32', name='conv_turn_2')
    t3 = Input(shape=(100,), dtype='int32', name='conv_turn_3')

    len1 = Input()
    len2 = Input()
    len3 = Input()

    case1 = Input()
    case2 = Input()
    case3 = Input()

    ########## Conversation layer, biggest ##########
    twitterEmbeddings = Embedding(embeddingMatrix.shape[0], EMBEDDING_DIM, weights=[embeddingMatrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    emb1 = twitterEmbeddings(t1)
    emb2 = twitterEmbeddings(t2)
    emb3 = twitterEmbeddings(t3)

    ################################################


    #LSTM layers, need to define a new one for different embeddings
    lstm = Bidirectional(LSTM(LSTM_DIM, dropout=DROPOUT, return_sequences=True))
    lstm_smiley = LSTM(LSTM_DIM, dropout=0.2)

    lstm1 = lstm(emb1)
    lstm2 = lstm(emb2)
    lstm3 = lstm(emb3)

    lstm4 = lstm_smiley(emb_smiley)

    #full network
    concatenated_lstm = Concatenate(axis=-1)([lstm1, lstm2, lstm3])
    reshaped_lstm = Flatten()(concatenated_lstm)

    concatenated_smiley = Concatenate(axis=-1)([reshaped_lstm, lstm4])

    #cool hidden
    hidden_layer = Dense(256, activation='relu')(concatenated_smiley)
    dropout = Dropout(0.2)(hidden_layer)

    #output
    model_output = Dense(4, activation='sigmoid')(dropout)
    model = Model([t1, t2, t3, len1, len2, len3, case1, case2, case3], model_output)

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    adam =  optimizers.adam(lr=LEARNING_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])

    return model

#' # Model Training + Evaluation

#+ models, echo=False
def main():
    print("Import new dataframe")
    #use new pandas dataframe with binary classification
    ind, labels, t1, t2, t3, len1, len2, len3, case1, case2, case3 = preprocessData(trainDataPath)
    ind_val, validationLabels, t1_val, t2_val, t3_val, len1_val, len2_val, len3_val, case1_val, case2_val, case3_val = preprocessData(validationDataPath)
    ind_test, testLabels, t1_test, t2_test, t3_test, len1_test, len2_test, len3_test, case1_test, case2_test, case3_test = preprocessData(testDataPath)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(t1+t2+t3)
    wordIndex = tokenizer.word_index

    t1, t2, t3 = tokenizer.texts_to_sequences(t1), tokenizer.texts_to_sequences(t2), tokenizer.texts_to_sequences(t3)
    t1_val, t2_val, t3_val = tokenizer.texts_to_sequences(t1_val), tokenizer.texts_to_sequences(t2_val), tokenizer.texts_to_sequences(t3_val)
    t1_test, t2_test, t3_test = tokenizer.texts_to_sequences(t1_test), tokenizer.texts_to_sequences(t2_test), tokenizer.texts_to_sequences(t3_test)

    print("Pad everything")
    t1, t2, t3 = pad_sequences(t1, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t2, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t3, maxlen=MAX_SEQUENCE_LENGTH)
    t1_val, t2_val, t3_val = pad_sequences(t1_val, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t2_val, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t3_val, maxlen=MAX_SEQUENCE_LENGTH)
    t1_test, t2_test, t3_test = pad_sequences(t1_test, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t2_test, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t3_test, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    validationLabels = to_categorical(np.asarray(validationLabels))
    testLabels = to_categorical(np.asarray(testLabels))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)
    smileyEmbeddings = getSmileyEmbeddings(wordIndex)
    print(embeddingMatrix.shape)
    print("Shape of training data tensor: ", u1_data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    np.random.shuffle(ind)
    t1 = t1[ind]
    t2 = t2[ind]
    t3 = t3[ind]
    case1 = [case1[i] for i in ind]
    case2 = [case2[i] for i in ind]
    case3 = [case3[i] for i in ind]
    len1 = [len1[i] for i in ind]
    len2 = [len2[i] for i in ind]
    len3 = [len3[i] for i in ind]
    labels = labels[ind]
    metrics = {"accuracy" : [], "microPrecision" : [], "microRecall" : [], "microF1" : []}

    print("Set up mo.. mod... mod..EL.. MODEL MOOODEEEL")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
    mc = ModelCheckpoint('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model = buildModel(embeddingMatrix, smileyEmbeddings)
    history = model.fit([t1, t2, t3, len1, len2, len3, case1, case2, case3], labels, validation_data=([t1_val, t2_val, t3_val, len1_val, len2_val, len3_val, case1_val, case2_val, case3_val], validationLabels), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[es, mc])

    print("Evaluating on Test Data...")
    model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    predictions = model.predict([t1_test, t2_test, t3_test, len1_test, len2_test, len3_test, case1_test, case2_test, case3_test], batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, testLabels)

    print("Creating solution file...")
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d"
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))

    if EARLY_STOPPING=='True':
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

if __name__ == '__main__':
    main()
