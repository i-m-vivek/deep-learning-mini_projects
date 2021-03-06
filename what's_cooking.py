import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

from google.colab import drive
drive.mount('/content/drive')

path_train = "/content/drive/My Drive/datasets/whats-cooking-kernels-only/train.json"
path_test = "/content/drive/My Drive/datasets/whats-cooking-kernels-only/test.json"

data_train = pd.read_json(path_train)
data_test = pd.read_json(path_test)

data_train.head()

data_test.head()

ingredients2index = {}
count = 0
for i in range(len(data_train)):
    for j in range(len(data_train["ingredients"][i])):
        if data_train["ingredients"][i][j] in ingredients2index:
            continue
        else :
            ingredients2index[data_train["ingredients"][i][j]] = count 
            count += 1

len(ingredients2index)

ingredients2index["UNK"] =  len(ingredients2index)

for i in range(len(data_train)):
    for j in range(len(data_train["ingredients"][i])):
        data_train["ingredients"][i][j] = ingredients2index[data_train["ingredients"][i][j]]

data_train.head()

cuisine2index = {}
count = 0
for i in range(len(data_train)):
    if data_train["cuisine"][i] in cuisine2index:
        continue
    else:
        cuisine2index[data_train["cuisine"][i]] = count 
        count = count + 1

cuisine2index["UNK"] = len(cuisine2index)

cuisine2index

data_train.head()

for i in range(len(data_train)):
    data_train["cuisine"][i] = cuisine2index[data_train["cuisine"][i]]

data_train.head()

"""Firstly we will create the word embedding for the ingredients. Then we will try diffrent methods to pass these embedding to the neural network.<br>

Figuring out how to create the supervised task to produce relevant representations is the toughest part of making embeddings.
"""

vocab_size = len(ingredients2index)
emb_dim = 100

x_train = data_train["ingredients"].values
y_train = data_train["cuisine"].values

y_train = to_categorical(y_train)

y_train.shape

max_v = 0
for i in x_train:
    if len(i) > max_v :
        max_v = len(i)

print(max_v)

x_train

y_train

print(x_train.shape, y_train.shape)

x_train_new = np.zeros(shape = (x_train.shape[0], 65), dtype=np.int16)

for i in range(len(x_train)):
    for j in range(len(x_train[i])):
        x_train_new[i][j] = x_train[i][j]

x_train_new

x_train_new.shape

y_train.shape

cuisine2index

len(cuisine2index)

model = Sequential()
model.add(Embedding(vocab_size, emb_dim, input_length=max_v))
model.add(Flatten())
model.add(Dense(20,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
model.fit(x_train_new, y_train, epochs=50,)

np.array(model.get_weights())[0].shape

data_test.head()

for i in range(len(data_test)):
    for j in range(len(data_test["ingredients"][i])):
        if data_test["ingredients"][i][j] not in ingredients2index:
            data_test["ingredients"][i][j] = ingredients2index["UNK"] 
        else: 
            data_test["ingredients"][i][j] = ingredients2index[data_test["ingredients"][i][j]]

data_test.head()

x_test = data_test["ingredients"]

x_test.shape

x_test_new = np.zeros(shape = (x_test.shape[0], 65), dtype=np.int16)

for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        x_test_new[i][j] = x_test[i][j]

x_test_new

pred = model.predict(x_test_new)

pred.shape

preds  = np.argmax(pred, axis = 1)

preds

