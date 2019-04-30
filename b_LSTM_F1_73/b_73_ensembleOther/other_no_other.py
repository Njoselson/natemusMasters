# %% # Smiley Separation Model
#' This is an improvement to our smiley separation
#' which now takes the 3 turns as separate inputs
#' into the LSTM layers.
#' rasmus created another config for exprementation

#+ setup, echo=False
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



global trainDataPath, testDataPath, solutionPath, gloveDir
global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE, EARLY_STOPPING

#parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
#parser.add_argument('-config', help='Config to read details', required=True)
#args = parser.parse_args()

with open('test.config') as configfile:
    config = json.load(configfile)

trainDataPath = config["train_data_path"]
validationDataPath = config["validation_data_path"]
testDataPath = config["test_data_path"]
solutionPath = config["solution_path"]
gloveDir = config["glove_dir"]

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

#### THIS IS FOR OTHER OR NOT OTHERS ####
def preprocessData2(dir):
    df = pd.read_csv(dir, sep='\t')

    df.label = df.label.apply(other_or_emo)
    labels = df.label.tolist()

    df['t1_len'] = df['turn1'].apply(count_length)
    df['t2_len'] = df['turn2'].apply(count_length)
    df['t3_len'] = df['turn3'].apply(count_length)

    df['t1_upper'] = df['turn1'].apply(count_upper_case)
    df['t2_upper'] = df['turn2'].apply(count_upper_case)
    df['t3_upper'] = df['turn3'].apply(count_upper_case)

    df['t1_smil'] = df['turn1'].apply(str2emoji).apply(lambda x: sum([1 for s in x if is_emoji(s)]))
    df['t2_smil'] = df['turn2'].apply(str2emoji).apply(lambda x: sum([1 for s in x if is_emoji(s)]))
    df['t3_smil'] = df['turn3'].apply(str2emoji).apply(lambda x: sum([1 for s in x if is_emoji(s)]))

    df['turn1'] = df['turn1'].apply(str2emoji).apply(remove_emoji).apply(lambda x: x.lower()).apply(preprocessString)
    df['turn2'] = df['turn2'].apply(str2emoji).apply(remove_emoji).apply(lambda x: x.lower()).apply(preprocessString)
    df['turn3'] = df['turn3'].apply(str2emoji).apply(remove_emoji).apply(lambda x: x.lower()).apply(preprocessString)

    ind = df.id.tolist()

    t1, t2, t3 = np.asarray(df.turn1), np.asarray(df.turn2), np.asarray(df.turn3)
    len1, len2, len3 = np.asarray(df.t1_len), np.asarray(df.t2_len), np.asarray(df.t3_len)
    smil1, smil2, smil3 = np.asarray(df.t1_smil), np.asarray(df.t2_smil), np.asarray(df.t3_smil)
    case1, case2, case3 = np.asarray(df.t1_upper), np.asarray(df.t2_upper), np.asarray(df.t3_upper)

    labels = to_categorical(np.asarray(labels))

    return ind, labels, t1, t2, t3, len1, len2, len3, case1, case2, case3, smil1, smil2, smil3
############################

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

#' # Model Specification
#' # An extra hidden layer which is supposed to extract the important information from the
#' # first embedding then add information from smilys

#+ model, echo=True
def buildModel(embeddingMatrix):

    ################## SECOND MODEL ###############################
    # did not define new hyperparams, many are hardcoded in model

    t1 = Input(shape=(100,), dtype='int32', name='turn_1')
    t2 = Input(shape=(100,), dtype='int32', name='turn_2')
    t3 = Input(shape=(100,), dtype='int32', name='turn_3')

    meta_data = Input(shape=(9,), dtype='float32', name='meta_input')

    #Simple meta data layer
    hidden_layer = Dense(16, activation='relu')
    meta_layer = hidden_layer(meta_data)

    #Embeddings
    secondEmbeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    emb1_secondModel = secondEmbeddingLayer(t1)
    emb2_secondModel = secondEmbeddingLayer(t2)
    emb3_secondModel = secondEmbeddingLayer(t3)

    #LSTM layers, need to define a new one for different embeddings
    lstm = Bidirectional(LSTM(128, dropout=DROPOUT, return_sequences=True))
    second_layer = LSTM(32, dropout=DROPOUT)

    lstm1_secondModel = lstm(emb1_secondModel)
    lstm2_secondModel = lstm(emb2_secondModel)
    lstm3_secondModel = lstm(emb3_secondModel)

    concatenated_lstm2 = Concatenate(axis=-1)([lstm1_secondModel, lstm2_secondModel, lstm3_secondModel])
    concatenated_lstm2 = Dropout(DROPOUT)(concatenated_lstm2)
    concatenated_lstm2 = second_layer(concatenated_lstm2)
    concatenated_lstm2 = Dropout(DROPOUT)(concatenated_lstm2)

    #merge with meta data
    merge_out = Concatenate(axis=-1)([concatenated_lstm2, meta_layer])
    dropout2 = Dropout(DROPOUT)(merge_out)

    #Output model 2
    model2_output = Dense(2, activation='sigmoid')(dropout2)

    ###############################################################

    #define model
    model = Model([t1, t2, t3, meta_data], model2_output)

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    adam =  optimizers.adam(lr=LEARNING_RATE)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    return model



#' # Model Training + Evaluation

#+ models, echo=False
def main():
    ##### PREPROCESSING NO 2, FOR OTHERS NOT OTHER MODEL #####
    trainIndices, labels2, t1, t2, t3, len1, len2, len3, case1, case2, case3, smil1, smil2, smil3 = preprocessData2(trainDataPath)
    ind_val, labels2_val, t1_val, t2_val, t3_val, len1_val, len2_val, len3_val, case1_val, case2_val, case3_val, smil1_val, smil2_val, smil3_val = preprocessData2(validationDataPath)
    ind_test, labels2_test, t1_test, t2_test, t3_test, len1_test, len2_test, len3_test, case1_test, case2_test, case3_test, smil1_test, smil2_test, smil3_test = preprocessData2(testDataPath)

    print("Extracting tokens for second model...")
    tokenizerModelTwo = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizerModelTwo.fit_on_texts(t1+t2+t3)
    wordIndexModelTwo = tokenizerModelTwo.word_index

    t1, t2, t3 = tokenizerModelTwo.texts_to_sequences(t1), tokenizerModelTwo.texts_to_sequences(t2), tokenizerModelTwo.texts_to_sequences(t3)
    t1_val, t2_val, t3_val = tokenizerModelTwo.texts_to_sequences(t1_val), tokenizerModelTwo.texts_to_sequences(t2_val), tokenizerModelTwo.texts_to_sequences(t3_val)
    t1_test, t2_test, t3_test = tokenizerModelTwo.texts_to_sequences(t1_test), tokenizerModelTwo.texts_to_sequences(t2_test), tokenizerModelTwo.texts_to_sequences(t3_test)

    print("Populating second embedding matrix")
    model2embeddings = getEmbeddingMatrix(wordIndexModelTwo)

    print("Pad everything")
    t1, t2, t3 = pad_sequences(t1, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t2, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t3, maxlen=MAX_SEQUENCE_LENGTH)
    t1_val, t2_val, t3_val = pad_sequences(t1_val, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t2_val, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t3_val, maxlen=MAX_SEQUENCE_LENGTH)
    t1_test, t2_test, t3_test = pad_sequences(t1_test, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t2_test, maxlen=MAX_SEQUENCE_LENGTH), pad_sequences(t3_test, maxlen=MAX_SEQUENCE_LENGTH)

    print("preprocess for SECOND model done")
    ##########################################################

    # Randomize data
    print("Shuffle all data (both model 1 and 2)")
    np.random.shuffle(trainIndices)

    t1 = t1[trainIndices]
    t2 = t2[trainIndices]
    t3 = t3[trainIndices]

    case1 = np.asarray([case1[i] for i in trainIndices])
    case2 = np.asarray([case2[i] for i in trainIndices])
    case3 = np.asarray([case3[i] for i in trainIndices])
    len1 = np.asarray([len1[i] for i in trainIndices])
    len2 = np.asarray([len2[i] for i in trainIndices])
    len3 = np.asarray([len3[i] for i in trainIndices])
    smil1 = np.asarray([smil1[i] for i in trainIndices])
    smil2 = np.asarray([smil2[i] for i in trainIndices])
    smil3 = np.asarray([smil3[i] for i in trainIndices])

    labels2 = labels2[trainIndices]

    meta_data = np.asarray([len1, len2, len3, case1, case2, case3, smil1, smil2, smil3]).T
    meta_data_val = np.asarray([len1_val, len2_val, len3_val, case1_val, case2_val, case3_val, smil1_val, smil2_val, smil3_val]).T
    meta_data_test = np.asarray([len1_test, len2_test, len3_test, case1_test, case2_test, case3_test, smil1_test, smil2_test, smil3_test]).T

    metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}


    if EARLY_STOPPING=='True':
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        mc = ModelCheckpoint('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE), monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        # build and save first model for ensemble

        ###### OTHERS NOT OTHERS MODEL #####
        secondModel = buildModel(model2embeddings)
        secondModel.save('2_cat_model.h5')
        history = secondModel.fit([t1, t2, t3, meta_data], labels2, validation_data=([t1_val, t2_val, t3_val, meta_data_val], labels2_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[es, mc])
        ####################################
    else:
        model = buildModel(embeddingMatrix, smileyEmbeddings)
        model.fit([u1_data,u2_data,u3_data, smiley_trial, t1, t2, t3, meta_data], labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        model.save('EP%d_LR%de-5_LDim%d_BS%d_test.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))


    print("Evaluating on Test Data...")
    model = load_model('EP%d_LR%de-5_LDim%d_BS%d_test.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    predictions = model.predict([t1_test, t2_test, t3_test, meta_data_test], batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, testLabels)

    preds = pd.DataFrame({'t1': t1_test, 't2': t2_test, 't3': t3_test, 'true val': labels2_test, 'predictions': predictions})
    preds.to_csv('other_no_other.csv', sep='\t')

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
