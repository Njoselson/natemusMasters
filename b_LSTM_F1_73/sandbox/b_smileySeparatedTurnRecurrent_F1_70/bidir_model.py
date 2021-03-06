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
import emoji
import json, argparse, os
import re
import io
import sys
sys.path.append(os.getcwd())
from helper_functions import *



global trainDataPath, testDataPath, solutionPath, gloveDir
global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, MAX_SMILEY_LENGTH
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
MAX_SMILEY_LENGTH = config["smiley_max_len"]
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

def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    u1 = []
    u2 = []
    u3 = []
    smil_1 = []
    smil_2 = []
    smil_3 = []
    smileys = []

    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4])

            #Replace non-unicode smilys with unicode
            conv = str2emoji(conv)

            #Separate smilys w unicode
            conv = add_space(conv)

            #Many of the words not in embeddings are problematic due to apostrophes e.g. didn't
            conv = fix_apos(conv)

            #Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            #Find all smilys as array of strings
            row_smileys = remove_text(conv)

            #Do the same operations for each turn
            u1_line = conv.split(' <eos> ')[0]
            u2_line = conv.split(' <eos> ')[1]
            u3_line = conv.split(' <eos> ')[2]

            #separate smilys per turn
            s1 = remove_text(u1_line)
            s2 = remove_text(u2_line)
            s3 = remove_text(u3_line)

            smil_1.append(s1)
            smil_2.append(s2)
            smil_3.append(s3)

            u1.append(re.sub(duplicateSpacePattern, ' ', u1_line.lower()))
            u2.append(re.sub(duplicateSpacePattern, ' ', u2_line.lower()))
            u3.append(re.sub(duplicateSpacePattern, ' ', u3_line.lower()))
            smileys.append(row_smileys)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels, u1, u2, u3, smileys, smil_1, smil_2, smil_3
    else:
        return indices, conversations, u1, u2, u3, smileys


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
def buildModel(embeddingMatrix, smileyEmbeddings):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : three layer lstm model with one layer of meta data
    """

    x1 = Input(shape=(100,), dtype='int32', name='main_input1')
    x2 = Input(shape=(100,), dtype='int32', name='main_input2')
    x3 = Input(shape=(100,), dtype='int32', name='main_input3')
    smiley_1 = Input(shape=(100,), dtype='int32', name='main_input4')
    smiley_2 = Input(shape=(100,), dtype='int32', name='main_input5')
    smiley_3 = Input(shape=(100,), dtype='int32', name='main_input6')

    #pretrained embedding layers
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    smileyEmbeddingLayer = Embedding(smileyEmbeddings.shape[0], 300, weights=[smileyEmbeddings], input_length=MAX_SMILEY_LENGTH, trainable=True)

    emb1 = embeddingLayer(x1)
    emb2 = embeddingLayer(x2)
    emb3 = embeddingLayer(x3)

    emb1 = Dropout(DROPOUT)(emb1)
    emb2 = Dropout(DROPOUT)(emb2)
    emb3 = Dropout(DROPOUT)(emb3)

    #smiley embeddings
    smiley_emb1 = smileyEmbeddingLayer(smiley_1)
    smiley_emb2 = smileyEmbeddingLayer(smiley_2)
    smiley_emb3 = smileyEmbeddingLayer(smiley_3)

    smiley_emb1 = Dropout(DROPOUT)(smiley_emb1)
    smiley_emb2 = Dropout(DROPOUT)(smiley_emb2)
    smiley_emb3 = Dropout(DROPOUT)(smiley_emb3)

    #LSTM layers, need to define a new one for different embeddings
    lstm_person1 = Bidirectional(LSTM(LSTM_DIM,recurrent_dropout=0.5, dropout=DROPOUT, return_sequences=True))
    lstm_person2 = Bidirectional(LSTM(LSTM_DIM,recurrent_dropout=0.5, dropout=DROPOUT, return_sequences=True))
    lstm_smiley_person1 = LSTM(int(LSTM_DIM/2),recurrent_dropout=0.5, dropout=DROPOUT)
    lstm_smiley_person2 = LSTM(int(LSTM_DIM/2),recurrent_dropout=0.5, dropout=DROPOUT)

    lstm1 = lstm_person1(emb1)
    lstm2 = lstm_person2(emb2)
    lstm3 = lstm_person1(emb3)

    lstm4 = lstm_smiley_person1(smiley_emb1)
    lstm5 = lstm_smiley_person2(smiley_emb2)
    lstm6 = lstm_smiley_person1(smiley_emb3)

    #full network
    concatenated_lstm = Concatenate(axis=-1)([lstm1, lstm2, lstm3])
    reshaped_lstm = Flatten()(concatenated_lstm)

    concatenated_smiley = Concatenate(axis=-1)([reshaped_lstm, lstm4, lstm5, lstm6])
    dropout = Dropout(0.2)(concatenated_smiley)

    #cool hidden
    hidden_layer = Dense(256, activation='relu')(concatenated_smiley)
    dropout = Dropout(0.2)(hidden_layer)

    #output
    model_output = Dense(4, activation='sigmoid')(dropout)
    model = Model([x1, x2, x3, smiley_1, smiley_2, smiley_3], model_output)

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    adam =  optimizers.adam(lr=LEARNING_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])

    return model

#' # Model Training + Evaluation

#+ models, echo=False
def main():
    print("Processing training data...")
    trainIndices, trainTexts, labels, u1_train, u2_train, u3_train, smil_train, s1_train, s2_train, s3_train = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing validation data...")
    validationIndices, validationTexts, validationLabels, u1_val, u2_val, u3_val, smil_val, s1_val, s2_val, s3_val = preprocessData(validationDataPath, mode="train")
    # writeNormalisedData(testDataPath, testTexts)
    print("Processing test data...")
    testIndices, testTexts, testLabels, u1_test, u2_test, u3_test, smil_test, s1_test, s2_test, s3_test = preprocessData(testDataPath, mode="train")

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(u1_train+u2_train+u3_train)
    u1_trainSequences, u2_trainSequences, u3_trainSequences, smil_trainSeq = tokenizer.texts_to_sequences(u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train), tokenizer.texts_to_sequences(smil_train)
    u1_valSequences, u2_valSequences, u3_valSequences, smil_valSeq = tokenizer.texts_to_sequences(u1_val), tokenizer.texts_to_sequences(u2_val), tokenizer.texts_to_sequences(u3_val), tokenizer.texts_to_sequences(smil_val)
    u1_testSequences, u2_testSequences, u3_testSequences, smil_testSeq = tokenizer.texts_to_sequences(u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test), tokenizer.texts_to_sequences(smil_test)

    s1_train_seq, s2_train_seq, s3_train_seq = tokenizer.texts_to_sequences(s1_train), tokenizer.texts_to_sequences(s2_train), tokenizer.texts_to_sequences(s3_train)
    s1_val_seq, s2_val_seq, s3_val_seq = tokenizer.texts_to_sequences(s1_val), tokenizer.texts_to_sequences(s2_val), tokenizer.texts_to_sequences(s3_val)
    s1_test_seq, s2_test_seq, s3_test_seq = tokenizer.texts_to_sequences(s1_test), tokenizer.texts_to_sequences(s2_test), tokenizer.texts_to_sequences(s3_test)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)
    smileyEmbeddings = getSmileyEmbeddings(wordIndex)

    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    s1_data = pad_sequences(s1_train_seq, maxlen=MAX_SMILEY_LENGTH)
    s2_data = pad_sequences(s2_train_seq, maxlen=MAX_SMILEY_LENGTH)
    s3_data = pad_sequences(s3_train_seq, maxlen=MAX_SMILEY_LENGTH)
    smil_data = pad_sequences(smil_trainSeq, maxlen=20)
    labels = to_categorical(np.asarray(labels))

    u1_valData = pad_sequences(u1_valSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_valData = pad_sequences(u2_valSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_valData = pad_sequences(u3_valSequences, maxlen=MAX_SEQUENCE_LENGTH)

    s1_valData = pad_sequences(s1_val_seq, maxlen=MAX_SMILEY_LENGTH)
    s2_valData = pad_sequences(s1_val_seq, maxlen=MAX_SMILEY_LENGTH)
    s3_valData = pad_sequences(s1_val_seq, maxlen=MAX_SMILEY_LENGTH)
    smil_valData = pad_sequences(smil_valSeq, maxlen=20)

    validationLabels = to_categorical(np.asarray(validationLabels))
    u1_testData = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_testData = pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_testData = pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)

    s1_testData = pad_sequences(s1_test_seq, maxlen=MAX_SMILEY_LENGTH)
    s2_testData = pad_sequences(s2_test_seq, maxlen=MAX_SMILEY_LENGTH)
    s3_testData = pad_sequences(s3_test_seq, maxlen=MAX_SMILEY_LENGTH)
    smil_testData = pad_sequences(smil_testSeq, maxlen=20)
    testLabels = to_categorical(np.asarray(testLabels))
    print("Shape of training data tensor: ", u1_data.shape)
    print("Shape of training data tensor: ", s1_data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    np.random.shuffle(trainIndices)
    u1_data = u1_data[trainIndices]
    u2_data = u2_data[trainIndices]
    u3_data = u3_data[trainIndices]
    s1_data = s1_data[trainIndices]
    s2_data = s2_data[trainIndices]
    s3_data = s3_data[trainIndices]
    labels = labels[trainIndices]
    metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}


    smiley_test = smil_valData
    smiley_trial = smil_data[trainIndices]

    if EARLY_STOPPING=='True':
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        mc = ModelCheckpoint('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # fit model
        model = buildModel(embeddingMatrix, smileyEmbeddings)
        history = model.fit([u1_data, u2_data, u3_data, s1_data, s2_data, s3_data], labels, validation_data=([u1_valData,u2_valData,u3_valData, s1_valData, s2_valData, s3_valData], validationLabels), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[es, mc])
    else:
        model = buildModel(embeddingMatrix, smileyEmbeddings)
        model.fit([u1_data,u2_data,u3_data, s1_data, s2_data, s3_data], labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))


    print("Evaluating on Test Data...")
    model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    predictions = model.predict([u1_testData, u2_testData, u3_testData, s1_testData, s2_testData, s3_testData], batch_size=BATCH_SIZE)
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
