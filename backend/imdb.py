import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from numpy import array

#turn text->token id 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#to create layers of the model, apply activation functions, create dense and dropout layers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, LSTM, Conv1D

#divide dataset into train and test sets, and evaluate model performance
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('IMDB_Dataset.csv', names=['Review', 'Sentiment'], skiprows=1)
print("\n Sample of Raw Dataset:\n")
print(df.sample(10).to_string(index=False))
df = df[['Review', 'Sentiment']].dropna()

df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.capitalize()

#remove anyhting that is not positve or negstive
df = df[df['Sentiment'].isin(['Positive', 'Negative'])]

# Update texts AFTER cleaning df
texts = df['Review'].astype(str).values

# Encode labels
labels = df['Sentiment'].map({'Positive': 1, 'Negative': 0}).astype(int).values


# Tokenize and Pad
vocab_size = 50000
maxlen = 500
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=maxlen)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2, random_state=42)

print("\nSample Preprocessed Data for LSTM Model:\n")

# Display first 9 examples
for i in range(9):
    print(f"Review {i+1}:")
    print(f"Original Text: {texts[i][:150]}")
    print(f"Tokenized Sequence (first 10 tokens): {sequences[i][:10]}")
    print(f"Padded Sequence (first 10 values):    {padded[i][:10]}")

    if labels[i] == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    print(f"Label (Encoded): {labels[i]} ({sentiment})")
    print("-" * 80)

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=maxlen))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, maxlen))
model.summary()

history = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.2) 


loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# import matplotlib.pyplot as plt        

# train_acc = history.history['accuracy']        # list of accuracy per epoch
# test_acc = accuracy                            # final test accuracy

# epochs = range(1, len(train_acc) + 1)

# plt.figure(figsize=(8,5))

# # Training accuracy line
# plt.plot(epochs, train_acc, marker='o', label='Training Accuracy')

# # Test accuracy horizontal line
# plt.axhline(y=test_acc, color='red', linestyle='-', label='Test Accuracy')

# plt.title("Training Accuracy vs Test Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.ylim(0, 1.2)

# plt.legend()
# plt.grid(True)
# plt.show()

import pickle

# Save the model
model.save('imdb_sentiment_model.h5')

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Model and tokenizer saved successfully!")