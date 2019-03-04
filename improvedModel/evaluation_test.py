from improved_model import *

print("Processing training data...")
_, trainTexts, _, u1_train, u2_train, u3_train = preprocessData(trainDataPath, mode="train")
# Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
# writeNormalisedData(trainDataPath, trainTexts)
print("Processing test data...")
testIndices, devTexts, labels, u1, u2, u3 = preprocessData(testDataPath, mode="train")
# writeNormalisedData(testDataPath, testTexts)

print("get our awesome meta data that is super important...")
meta_data = getMetaData(testDataPath)

print("Extract tokens...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(u1_train + u2_train + u3_train)
u1_testSequences = tokenizer.texts_to_sequences(u1)
u2_testSequences = tokenizer.texts_to_sequences(u2)
u3_testSequences = tokenizer.texts_to_sequences(u3)

u1_data = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
u2_data = pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
u3_data = pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)

print("Loading model...")
model = load_model('EP5_LR300e-5_LDim128_BS200_imp.h5')

print("Make predictions...")
predictions = model.predict([u1_data, u2_data, u3_data, meta_data], batch_size=BATCH_SIZE)
labels = to_categorical(np.array(labels))

accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, labels)
