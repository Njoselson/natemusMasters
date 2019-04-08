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
from keras.layers import Dense, Embedding, LSTM, Input, Concatenate, Dropout, Bidirectional, Reshape, Flatten, TimeDistributed, Conv2D, Conv1D, MaxPooling1D, concatenate, merge
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
global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE, EARLY_STOPPING
# For char embeddings
global CHAR_MAX_LEN

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
CHAR_MAX_LEN = config["padding_characters"]
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
    smileys = []
    raw1 = []
    raw2 = []
    raw3 = []

    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure

            # Char embeddings in
            fix_emoji_line = str2emoji(line)
            splitted_line = fix_emoji_line.strip().split('\t')

            raw1.append(splitted_line[1])
            raw2.append(splitted_line[2])
            raw3.append(splitted_line[3])

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

            u1.append(re.sub(duplicateSpacePattern, ' ', u1_line.lower()))
            u2.append(re.sub(duplicateSpacePattern, ' ', u2_line.lower()))
            u3.append(re.sub(duplicateSpacePattern, ' ', u3_line.lower()))
            smileys.append(row_smileys)

            indices.append(int(line[0]))
            conversations.append(conv.lower())

    if mode == "train":
        return indices, conversations, labels, u1, u2, u3, smileys, raw1, raw2, raw3
    else:
        return indices, conversations, u1, u2, u3, smileys, raw1, raw2, raw3


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

####### THIS IS FOR THE CHARACTERS
def padded_char_vectors(u1, u2, u3):
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=False)
    tk.fit_on_texts(u1+u2+u3)
    alphabet = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    t1 = {}
    t2 = {}
    t3 = {}
    word_list_t1 = []
    word_list_t2 = []
    word_list_t3 = []
    padded_seq_t1 = []
    padded_seq_t2 = []
    padded_seq_t3 = []

    #keras stuff for fixing dict
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    tk.word_index = char_dict.copy()
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

    #crazy stupid looping to get everything on the right shape
    #for text_to_sequences you want a list of lists of each letter per word per sentence
    for k, sentence in enumerate(u1):
        t1[k] = sentence.split(' ')
        t1[k] = [list(word) for word in t1[k]]
    for k, sentence in enumerate(u2):
        t2[k] = sentence.split(' ')
        t2[k] = [list(word) for word in t2[k]]
    for k, sentence in enumerate(u3):
        t3[k] = sentence.split(' ')
        t3[k] = [list(word) for word in t3[k]]

    #remaking to list of list
    for key in t1:
        word_list_t1.append(t1[key])
    for key in t2:
        word_list_t2.append(t2[key])
    for key in t3:
        word_list_t3.append(t3[key])

    #pad the right things
    for word_per_turn in word_list_t1:
        seq_t1 = tk.texts_to_sequences(word_per_turn)
        padded1 = pad_sequences(seq_t1, maxlen=CHAR_MAX_LEN)
        padded_seq_t1.append(pad_array(padded1, CHAR_MAX_LEN))

    for word_per_turn in word_list_t2:
        seq_t2 = tk.texts_to_sequences(word_per_turn)
        padded2 = pad_sequences(seq_t2, maxlen=CHAR_MAX_LEN)
        padded_seq_t2.append(pad_array(padded2, CHAR_MAX_LEN))

    for word_per_turn in word_list_t3:
        seq_t3 = tk.texts_to_sequences(word_per_turn)
        padded3 = pad_sequences(seq_t3, maxlen=CHAR_MAX_LEN)
        padded_seq_t3.append(pad_array(padded3, CHAR_MAX_LEN))

    return np.asarray(padded_seq_t1), np.asarray(padded_seq_t2), np.asarray(padded_seq_t3)

    #speparate characters and pad sentences, compared to above that uses matrices in which each row contains
    #character representation for each word in a sentence
def pad_sentence(u1, u2, u3):
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=False)
    tk.fit_on_texts(u1+u2+u3)
    alphabet = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0gf123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    t1 = {}
    t2 = {}
    t3 = {}
    char_list_t1 = []
    char_list_t2 = []
    char_list_t3 = []
    padded_seq_t1 = []
    padded_seq_t2 = []
    padded_seq_t3 = []

    #keras stuff for fixing dict
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    tk.word_index = char_dict.copy()
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

    for sentence in u1:
        char_list_t1.append(list(sentence))
    char_list_t1 = tk.texts_to_sequences(char_list_t1)
    char_list_t1 = pad_sequences(char_list_t1, maxlen=CHAR_MAX_LEN)

    for sentence in u2:
        char_list_t2.append(list(sentence))
    char_list_t2 = tk.texts_to_sequences(char_list_t2)
    char_list_t2 = pad_sequences(char_list_t2, maxlen=CHAR_MAX_LEN)

    for sentence in u3:
        char_list_t3.append(list(sentence))
    char_list_t3 = tk.texts_to_sequences(char_list_t3)
    char_list_t3 = pad_sequences(char_list_t3, maxlen=CHAR_MAX_LEN)

    padded_seq_t1 = np.asarray(char_list_t1)
    padded_seq_t2 = np.asarray(char_list_t2)
    padded_seq_t3 = np.asarray(char_list_t3)

    return padded_seq_t1, padded_seq_t2, padded_seq_t3
    ####################


#' # Model Specification
#' # An extra hidden layer which is supposed to extract the important information from the
#' # first embedding then add information from smilys

#+ model, echo=True
def buildModel(embeddingMatrix, smileyEmbeddings):

    """ Constructs the architecture of the model """

    ################ CHARACTERS #########
    char_input1 = Input(shape=(CHAR_MAX_LEN,))
    char_input2 = Input(shape=(CHAR_MAX_LEN,))
    char_input3 = Input(shape=(CHAR_MAX_LEN,))

    conv = Conv1D(kernel_size=4, filters=30, padding='same', activation='tanh', strides=1)
    lstm_chars = Bidirectional(LSTM(LSTM_DIM*3,return_sequences=False, recurrent_dropout=0.25))

    emb_char1 = Embedding(99, 32, trainable=True)(char_input1)
    emb_char2 = Embedding(99, 32, trainable=True)(char_input2)
    emb_char3 = Embedding(99, 32, trainable=True)(char_input3)

    conv1 = conv(emb_char1)
    conv2 = conv(emb_char2)
    conv3 = conv(emb_char3)

    conc_conv = Concatenate(axis=-1)([conv1, conv2, conv3])
    lstm_char = lstm_chars(conc_conv)
    #######################################

    ###### Pretrained embeddings ##########
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    smileyEmbeddingLayer = Embedding(smileyEmbeddings.shape[0], 300, weights=[smileyEmbeddings], input_length=20, trainable=True)
    #######################################

    ######### Smiley LSTM #################
    #Normal lstm for smiley, doubt biderectional is better?
    smiley_input = Input(shape=(20,), dtype='int32', name='main_input4')
    lstm_smiley = LSTM(LSTM_DIM, recurrent_dropout=0.2)

    emb_smiley = smileyEmbeddingLayer(smiley_input)
    lstm_smil = lstm_smiley(emb_smiley)
    lstm_smil = Dropout(DROPOUT)(lstm_smil)
    ########################################

    ######### WORDS ########################
    x1 = Input(shape=(100,), dtype='int32', name='main_input1')
    x2 = Input(shape=(100,), dtype='int32', name='main_input2')
    x3 = Input(shape=(100,), dtype='int32', name='main_input3')
    #LSTM for words, need to define a new one for different embeddings
    lstm = Bidirectional(LSTM(LSTM_DIM, recurrent_dropout=0.2, return_sequences=True))
    lstm_cat = Bidirectional(LSTM(LSTM_DIM, recurrent_dropout=0.2, return_sequences=True))

    emb1 = embeddingLayer(x1)
    emb2 = embeddingLayer(x2)
    emb3 = embeddingLayer(x3)

    lstm1 = lstm(emb1)
    lstm2 = lstm(emb2)
    lstm3 = lstm(emb3)

    concatenated_word_lstm = Concatenate(axis=-1)([lstm1, lstm2, lstm3])
    concatenated_word_lstm = Flatten()(concatenated_word_lstm)
    concatenated_word_lstm = Dropout(DROPOUT)(concatenated_word_lstm)
    ########################################


    ####### PUTTING IT ALL TOGETHER ########
    #awesome word-smil-char concatenation
    concatenated_smiley_char = Concatenate(axis=-1)([concatenated_word_lstm, lstm_smil, lstm_char])
    concatenated_smiley_char = Dropout(DROPOUT)(concatenated_smiley_char)

    #New Hidden layer to weigh everything together
    concatenated_smiley_char = Dense(128, activation='relu')(concatenated_smiley_char)
    concatenated_smiley_char = Dropout(DROPOUT)(concatenated_smiley_char)

    #output
    model_output = Dense(4, activation='sigmoid')(concatenated_smiley_char)
    model = Model([x1, x2, x3, smiley_input, char_input1, char_input2, char_input3], model_output)
    #####################################


    ### THIS IS WHERE WE CAN FIX BERT ###

    #####################################

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    adam = optimizers.adam(lr=LEARNING_RATE)
    Nadam = optimizers.Nadam(lr=LEARNING_RATE)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Nadam,
                  metrics=['acc'])

    return model

#' # Model Training + Evaluation

#+ models, echo=False
def main():
    print("Processing training data...")
    trainIndices, trainTexts, labels, u1_train, u2_train, u3_train, smil_train, r1_train, r2_train, r3_train = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing validation data...")
    validationIndices, validationTexts, validationLabels, u1_val, u2_val, u3_val, smil_val, r1_val, r2_val, r3_val = preprocessData(validationDataPath, mode="train")
    # writeNormalisedData(testDataPath, testTexts)
    print("Processing test data...")
    testIndices, testTexts, testLabels, u1_test, u2_test, u3_test, smil_test, r1_test, r2_test, r3_test = preprocessData(testDataPath, mode="train")

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(u1_train+u2_train+u3_train)
    u1_trainSequences, u2_trainSequences, u3_trainSequences, smil_trainSeq = tokenizer.texts_to_sequences(u1_train), tokenizer.texts_to_sequences(u2_train), tokenizer.texts_to_sequences(u3_train), tokenizer.texts_to_sequences(smil_train)
    u1_valSequences, u2_valSequences, u3_valSequences, smil_valSeq = tokenizer.texts_to_sequences(u1_val), tokenizer.texts_to_sequences(u2_val), tokenizer.texts_to_sequences(u3_val), tokenizer.texts_to_sequences(smil_val)
    u1_testSequences, u2_testSequences, u3_testSequences, smil_testSeq = tokenizer.texts_to_sequences(u1_test), tokenizer.texts_to_sequences(u2_test), tokenizer.texts_to_sequences(u3_test), tokenizer.texts_to_sequences(smil_test)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Sorting char matrices")
    r1_train, r2_train, r3_train = pad_sentence(r1_train, r2_train, r3_train)
    r1_val, r2_val, r3_val = pad_sentence(r1_val, r2_val, r3_val)
    r1_test, r2_test, r3_test = pad_sentence(r1_test, r2_test, r3_test)

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)
    #embeddingMatrix = np.zeros((14613,200))
    smileyEmbeddings = getSmileyEmbeddings(wordIndex)
    #smileyEmbeddings = np.zeros((14613,300))
    print("twitter shape: ")
    print(embeddingMatrix.shape)
    print("smilemb shape: ")
    print(smileyEmbeddings.shape)

    u1_data = pad_sequences(u1_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_data = pad_sequences(u2_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_data = pad_sequences(u3_trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    smil_data = pad_sequences(smil_trainSeq, maxlen=20)
    labels = to_categorical(np.asarray(labels))
    u1_valData = pad_sequences(u1_valSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_valData = pad_sequences(u2_valSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_valData = pad_sequences(u3_valSequences, maxlen=MAX_SEQUENCE_LENGTH)
    smil_valData = pad_sequences(smil_valSeq, maxlen=20)
    validationLabels = to_categorical(np.asarray(validationLabels))
    u1_testData = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u2_testData = pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    u3_testData = pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    smil_testData = pad_sequences(smil_testSeq, maxlen=20)
    testLabels = to_categorical(np.asarray(testLabels))
    print("Shape of training data tensor: ", u1_data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    np.random.shuffle(trainIndices)
    u1_data = u1_data[trainIndices]
    u2_data = u2_data[trainIndices]
    u3_data = u3_data[trainIndices]
    labels = labels[trainIndices]
    metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}


    smiley_test = smil_valData
    smiley_trial = smil_data[trainIndices]

    if EARLY_STOPPING=='True':
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        mc = ModelCheckpoint('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        # fit model
        model = buildModel(embeddingMatrix, smileyEmbeddings)
        print(model.summary())
        history = model.fit([u1_data,u2_data,u3_data, smiley_trial, r1_train, r2_train, r3_train], labels, validation_data=([u1_valData,u2_valData,u3_valData, smiley_test, r1_val, r2_val, r3_val], validationLabels), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2, callbacks=[es, mc])
    else:
        model = buildModel(embeddingMatrix, smileyEmbeddings)
        model.fit([u1_data,u2_data,u3_data, smiley_trial], labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
        model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))


    print("Evaluating on Test Data...")
    model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    predictions = model.predict([u1_testData, u2_testData, u3_testData, smil_testData, r1_test, r2_test, r3_test], batch_size=BATCH_SIZE)
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
