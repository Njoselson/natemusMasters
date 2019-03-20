from improved_model import *

print("Processing training data...")
_, trainTexts, _, u1_train, u2_train, u3_train, _ = preprocessData(trainDataPath, mode="train")
# Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
# writeNormalisedData(trainDataPath, trainTexts)
print("Processing test data...")
testIndices, devTexts, labels, u1, u2, u3, smil_test = preprocessData(testDataPath, mode="train")
# writeNormalisedData(testDataPath, testTexts)

print("get our awesome meta data that is super important...")
meta_data = getMetaData(testDataPath)

print("Extract tokens...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(u1_train + u2_train + u3_train)
u1_testSequences, u2_testSequences, u3_testSequences, smil_testSeq = tokenizer.texts_to_sequences(u1), tokenizer.texts_to_sequences(u2), tokenizer.texts_to_sequences(u3), tokenizer.texts_to_sequences(smil_test)

u1_data = pad_sequences(u1_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
u2_data = pad_sequences(u2_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
u3_data = pad_sequences(u3_testSequences, maxlen=MAX_SEQUENCE_LENGTH)
smiley_data = pad_sequences(smil_testSeq, maxlen=20)
labels = to_categorical(np.asarray(labels))

print("Loading model...")
model = load_model('EP5_LR200e-5_LDim128_BS200_SMIL.h5')

print("Make predictions...")
predictions = model.predict([u1_data, u2_data, u3_data, smiley_data], batch_size=BATCH_SIZE)

accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, labels)
