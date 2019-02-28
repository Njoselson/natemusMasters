from improved_model import *

print("Processing training data...")
_, trainTexts, _ = preprocessData(trainDataPath, mode="train")
# Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable
# writeNormalisedData(trainDataPath, trainTexts)
print("Processing test data...")
testIndices, devTexts, labels = preprocessData(testDataPath, mode="train")
# writeNormalisedData(testDataPath, testTexts)

print("Extract tokens...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(trainTexts)
devSequences = tokenizer.texts_to_sequences(devTexts)
data = pad_sequences(devSequences, maxlen=MAX_SEQUENCE_LENGTH)

print("Loading model...")
model = load_model('EP5_LR300e-5_LDim128_BS200.h5')

print("Make predictions...")
predictions = model.predict(data, batch_size=BATCH_SIZE)
labels = to_categorical(np.array(labels))

accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, labels)
