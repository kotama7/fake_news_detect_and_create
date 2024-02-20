import MeCab
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import torch
import tqdm
from transformers import (
    BertModel, BertJapaneseTokenizer
)
import unidic
import os
from keras import backend as K

class Blocker(tf.keras.Model):

    def __init__(self):
        super(Blocker, self).__init__()
        self.input_dim = 512
        self.division_expression = 768
        self.second_layer = 64
        self.third_layer = 8
        self.forth_layer = 1
        self.dense_1 = Dense(self.input_dim, activation="relu")
        self.dense_2 = Dense(self.second_layer, activation="relu")
        self.dense_3 = Dense(self.third_layer, activation="relu")
        self.dense_4 = Dense(self.forth_layer, activation="sigmoid")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    def call(self, x_input):
        hidden = tf.reshape(x_input, (-1, self.input_dim * self.division_expression))
        hidden = self.dense_1(hidden)
        hidden = self.dense_2(hidden)
        hidden = self.dense_3(hidden)
        hidden = self.dense_4(hidden)
        return hidden

    def preprocessing(self, text_ls, tokenizer, embedder):

        # data_size = len(text_ls)
        text_length = 512

        parser = MeCab.Tagger(f"-d {repr(unidic.DICDIR)} -Owakati")

        parsed_text_ls = [parser.parse(news) for news in text_ls]

        inputs = tokenizer.batch_encode_plus(parsed_text_ls, return_tensors="pt", padding="max_length", max_length=text_length, truncation=True , add_special_tokens=True)["input_ids"].to(self.device) 

        last_hidden_statesPre = torch.Tensor().to(self.device)
        print(inputs.shape)

        embedder = embedder.to(self.device)

        embedder.eval()

        for inid in tqdm.tqdm(inputs[:]):
            with torch.no_grad():  # 勾配計算なし
                tensor = inid.reshape(1,-1)
                attention_mask = torch.tensor([[int(i > 0) for i in tensor[0]]]).to(self.device)
                outputs = embedder(inid.reshape(1,-1), output_hidden_states=True, attention_mask=attention_mask)
            last_hidden_statesPre = torch.cat((last_hidden_statesPre, outputs.last_hidden_state))
        # 最終層の隠れ状態ベクトルを取得
        print(last_hidden_statesPre.shape)
        # 最後の隠れ層ベクトルsave
        return last_hidden_statesPre.to('cpu').detach().numpy().copy()
    

    def score(self, x, y):
        
        input_dim = 512
        vector_size = 768

        shaped_x = tf.reshape(x, (-1, input_dim * vector_size))

        pred_y = self(shaped_x)

        pred_y = tf.where(pred_y >= 0.5, 1, 0)

        # 2つのテンソルの要素ごとの一致を比較します。
        matches = tf.equal(y, pred_y)

        # 一致している要素の割合を計算します。
        matching_ratio = tf.reduce_mean(tf.cast(matches, dtype=tf.float32))
        
        return matching_ratio


@tf.function
def train_step(x,y):

    loss = 0
    input_dim = 512
    vector_size = 768

    shaped_x = tf.reshape(x, (-1, input_dim * vector_size))
    with tf.GradientTape() as tape:
        pred_y = blocker(shaped_x, training=True)
        entoropy = tf.keras.losses.binary_crossentropy(pred_y, y)
        loss += K.sum(entoropy)
    batch_loss = (loss / len(x))
    variables = blocker.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    # accuracyの計算用
    return batch_loss


if __name__ == "__main__":

    pretrained_source = u"cl-tohoku/bert-base-japanese"

    if not os.path.exists("./dataset/fake_X_data.npy"):
        
        tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_source)
        embedder = BertModel.from_pretrained(pretrained_source)

        text_ls = pd.read_csv("./Japanese-Fakenews-Dataset/trainable_fake_news.csv")["0"].to_list()

        blocker = Blocker()

        X_fake = blocker.preprocessing(text_ls,tokenizer,embedder)

        with open("./dataset/fake_X_data.npy", "wb") as f:
            np.save(f, X_fake)

    X_true = np.load("./dataset/X_data.npy")

    X_fake = np.load("./dataset/fake_X_data.npy")

    print("data_loaded!!")


    X_train, _ =  train_test_split(X_true, test_size=0.4, random_state=428)

    one_array = np.ones((len(X_train)))
    zero_array = np.zeros((len(X_fake)))

    X = np.vstack((X_train, X_fake))

    X = shuffle(X, random_state=428)

    Y = np.hstack((one_array, zero_array))

    Y = shuffle(Y, random_state=428)

    dataset_X = tf.data.Dataset.from_tensor_slices(X)
    dataset_Y = tf.data.Dataset.from_tensor_slices(Y)

    print("data_splited!!")

    if not os.path.exists("./model/blocker"):
        blocker = Blocker()
    else:
        blocker = load_model("./model/blocker")

    optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)

    text_length = 512


    BATCH_SIZE = 16
    dataset_X = tf.data.Dataset.from_tensor_slices((X_train))
    dataset_X = dataset_X.batch(BATCH_SIZE, drop_remainder=True)

    steps_per_epoch = len(X_train) // BATCH_SIZE # 何個に分けるか

    EPOCHS = 10
    for epoch in range(EPOCHS):
        
        for x, y in zip(dataset_X, dataset_Y):
            
            batch_loss = train_step(x, y)

        print('Epoch {} Loss {}'.format(epoch + 1, batch_loss.numpy()))
        
        blocker.save("./model/blocker")
