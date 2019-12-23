from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, \
  Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

!wget -nc http://files.grouplens.org/datasets/movielens/ml-20m.zip

!unzip -n ml-20m.zip

df = pd.read_csv('ml-20m/ratings.csv')
df.head()

df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes

df.movieId = pd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes

user_ids = df['new_user_id'].values
movie_ids = df['new_movie_id'].values
ratings = df['rating'].values

N = len(set(user_ids))
M = len(set(movie_ids))

K = 10

u = Input(shape=(1,))

m = Input(shape=(1,))

u_emb = Embedding(N, K)(u) # (num_samples, 1, K)

m_emb = Embedding(M, K)(m) # (num_samples, 1, K)

u_emb = Flatten()(u_emb) # (num_samples, K)
m_emb = Flatten()(m_emb) # (num_samples, K)

x = Concatenate()([u_emb, m_emb]) # (num_samples, 2K)

x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(1)(x)

model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  optimizer="adam",
)

user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)
Ntrain = int(0.8 * len(ratings))
train_user = user_ids[:Ntrain]
train_movie = movie_ids[:Ntrain]
train_ratings = ratings[:Ntrain]

test_user = user_ids[Ntrain:]
test_movie = movie_ids[Ntrain:]
test_ratings = ratings[Ntrain:]

r = model.fit(
  x=[train_user, train_movie],
  y=train_ratings,
  epochs=50,
  batch_size=1024,
  verbose=2,
  validation_data=([test_user, test_movie], test_ratings),
)

plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="val loss")
plt.legend()
plt.show()

