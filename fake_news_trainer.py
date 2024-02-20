import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model
from keras import backend as K
from transformers import (
    BertModel, BertJapaneseTokenizer
)
import os

from sklearn.model_selection import train_test_split
import random

import torch
import tqdm
import pandas as pd
import MeCab
import numpy as np
import unidic
from copy import deepcopy
from DeEmbedder import DeEmbedder
import pickle

class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.input_dim = 512
        self.division_expression = 768
        hidden_dim_1 = 512
        self.latent_dim = 64

        self.dense_1 = Dense(hidden_dim_1, activation='relu')
        self.dense_mu = Dense(self.latent_dim, activation='linear')
        self.dense_log_sigma = Dense(self.latent_dim, activation='linear', kernel_initializer='zeros')


 
    def call(self, x_input):
        hidden = self.dense_1(x_input)
        mu = self.dense_mu(hidden)
        log_sigma = self.dense_log_sigma(hidden)
        eps = K.random_normal(shape=(self.latent_dim,), mean=0., stddev=0.1)
        z = mu + K.exp(log_sigma) * eps
       
        return mu, log_sigma, z


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.input_dim = 512
        self.vector_size = 768
        self.hidden_dim_1 = 128
        self.hidden_dim_2 = 128
        self.hidden_dim_3 = 128
        self.hidden_dim_1_layer = Dense(self.hidden_dim_1, activation = "relu")
        self.hidden_dim_2_layer = Dense(self.hidden_dim_2, activation = "relu")
        self.hidden_dim_3_layer = Dense(self.hidden_dim_3, activation = "relu")
        self.word_output = Dense(self.input_dim, activation = "relu")
        self.vector_output = Dense(self.input_dim * self.vector_size, activation = "linear")
    
    def call(self, z):
        output = self.word_output(z)
        output = self.hidden_dim_1_layer(output)
        output = self.hidden_dim_2_layer(output)
        output = self.hidden_dim_3_layer(output)
        output = self.vector_output(output)
        return output

class VAE(tf.keras.Model):
  
    def __init__(self):
        super(VAE, self).__init__()
        self.vocab_size = 32006
        self.input_dim = 512
        self.vector_size = 768
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


    def call(self, x):
        mu, log_sigma, z = self.encode(x)
        x_decoded = self.decode(z)
        return mu, log_sigma, x_decoded, z
    
    def encode(self, x):
        mu, log_sigma, z = self.encoder(x)
        return mu, log_sigma, z
    

    def decode(self, z):
        x_decoded = self.decoder(z)
        return x_decoded
    
    
    def preprocessing(self, text_ls, tokenizer, embedder):

        # data_size = len(text_ls)
        text_length = 512

        parser = MeCab.Tagger(f"-d {repr(unidic.DICDIR)} -Owakati")

        parsed_text_ls = [parser.parse(news) for news in text_ls]

        inputs = tokenizer.batch_encode_plus(parsed_text_ls, return_tensors="pt", padding="max_length", max_length=text_length, truncation=True , add_special_tokens=True)["input_ids"].to(self.device)
        # 分散表現抽出
        # 一気に処理するとMemory errorになるのでFor文で1行づつ抽出

        # Y_ls = np.zeros((data_size, text_length, self.vocab_size))
        
        # for i, text_ls in enumerate(inputs.tolist()):
        #     for j, word_id in enumerate(text_ls):
        #         Y_ls[i][j][int(word_id)] = 1

        Y_ls = inputs.to("cpu").detach().numpy().copy()

        last_hidden_statesPre = torch.Tensor().to(self.device)
        print(inputs.shape)

        embedder = embedder.to(self.device)

        embedder.eval()

        for inid in tqdm.tqdm(inputs[:]):
            with torch.no_grad():  # 勾配計算なし
                tensor = inid.reshape(1,-1)
                attention_mask = torch.tensor([[int(i > 0) for i in tensor[0]]]).to(self.device)
                outputs = embedder(inid.reshape(1,-1), output_hidden_states=True, attention_mask=attention_mask)
            last_hidden_statesPre = torch.cat((last_hidden_statesPre, outputs.last_hidden_state)).to(self.device)
        # 最終層の隠れ状態ベクトルを取得
        print(last_hidden_statesPre.shape)
        # 最後の隠れ層ベクトルsave
        return last_hidden_statesPre.to("cpu").detach().numpy().copy(), Y_ls
        # > torch.Size([6099, 258, 768])
        # > 2h 10min 46s


def reconstract(vae, array, tokenizer, deembedder):
    input_dim = 512
    vector_size = 768
    reshaped_array = tf.reshape(array, (-1, input_dim * vector_size))
    _, __, reshaped_array, ___ =  vae.predict(reshaped_array)
    reshaped_array = tf.reshape(reshaped_array, (-1, input_dim , vector_size))
    temp = np.zeros((len(reshaped_array), input_dim))
    for i in range(len(reshaped_array)):
            temp[i] = np.argmax(deembedder(reshaped_array[i]).numpy(), axis=1)

    text_ls = ["".join(text) for text in tokenizer.batch_decode(temp)]
    
    return text_ls    

@tf.function
def train_step(x):
    loss = 0
    input_dim = 512
    vector_size = 768
    shaped_x = tf.reshape(x, (-1, input_dim * vector_size))

    with tf.GradientTape() as tape:

        mu, log_sigma, x_reconstructed, z = vae(shaped_x, training=True)
        reconstruction_loss = tf.keras.metrics.mean_squared_error(shaped_x, x_reconstructed)
        kl_loss = -0.5 * (1 + log_sigma - K.square(mu) - K.exp(log_sigma))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        vae_loss = tf.reduce_mean(kl_loss)
        rc_loss = tf.reduce_mean(reconstruction_loss)
        loss += rc_loss + vae_loss
            
    batch_loss = loss / len(shaped_x)
    variables = vae.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    
    
    # accuracyの計算用
    return batch_loss


if __name__ == "__main__":

    pretrained_source = u"cl-tohoku/bert-base-japanese"

    if not os.path.exists("./dataset/X_data.npy"):
        dataset = pd.read_csv("./Japanese-Fakenews-Dataset/trainable_true_news.csv")
        news_list = dataset.iloc[:,1].to_list()
        tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_source)
        embedder = BertModel.from_pretrained(pretrained_source)
        vae = VAE()
        X, Y = vae.preprocessing(news_list, tokenizer, embedder)

        with open("./dataset/X_data.npy", "wb") as f:
            np.save(f, X)

        with open("./dataset/Y_data.npy", "wb") as f:
            np.save(f, Y)


    X = np.load("./dataset/X_data.npy")

    print("data_loaded!!")

    X_train, X_test =  train_test_split(X, test_size=0.4, random_state=428)

    print("data splited")


    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    if not os.path.exists("./model/vae"):
        vae = VAE()
    else:
        vae = load_model("./model/vae")

    BATCH_SIZE = 16
    dataset_X = tf.data.Dataset.from_tensor_slices((X_train))
    dataset_X = dataset_X.batch(BATCH_SIZE, drop_remainder=True)

    steps_per_epoch = len(X_train) // BATCH_SIZE # 何個に分けるか

    EPOCHS = 1000

    for epoch in range(EPOCHS):
        for batch, x in enumerate(dataset_X):
            
            batch_loss = train_step(x)

            if batch == 0:
                print(reconstract(vae, x[0:5], BertJapaneseTokenizer.from_pretrained(pretrained_source), load_model("./model/DeEmbedder")))

        print('Epoch {} Loss {}'.format(epoch + 1, batch_loss.numpy()))
        vae.save("./model/vae")



    #学習データ作成
    # with open("./dataset/X_data.npy", "wb") as f:
    #     np.save(f, X)

    # with open("./dataset/Y_data.npy", "wb") as f:
    #     np.save(f, Y)





    #不適切データ削除
    # text_ls = []

    # for text in news_list:
    #     if len(tokenizer(text, is_split_into_words=True, add_special_tokens=True)["input_ids"]) <= 512:
    #         text_ls.append(text)

    # pd.Series(text_ls).to_csv("./Japanese-Fakenews-Dataset/trainable_true_news.csv")