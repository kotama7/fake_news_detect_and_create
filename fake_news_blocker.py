import MeCab
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense
import torch
import tqdm
from transformers import (
    BertModel, BertJapaneseTokenizer
)
import unidic


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

        inputs = tokenizer.batch_encode_plus(parsed_text_ls, return_tensors="pt", padding="max_length", max_length=text_length, truncation=True , add_special_tokens=True)["input_ids"]

        last_hidden_statesPre = torch.Tensor()
        print(inputs.shape)

        embedder.eval()

        for inid in tqdm.tqdm(inputs[:]):
            with torch.no_grad():  # 勾配計算なし
                outputs = embedder(inid.reshape(1,-1), output_hidden_states=True)
            last_hidden_statesPre = torch.cat((last_hidden_statesPre, outputs.last_hidden_state))
        # 最終層の隠れ状態ベクトルを取得
        print(last_hidden_statesPre.shape)
        # 最後の隠れ層ベクトルsave
        return last_hidden_statesPre.to('cpu').detach().numpy().copy()
    
# @tf.function
# def train_step(x,y):
#     loss = 0
#     with tf.GradientTape() as tape:
#         ans = blocker(x, training=True)
#         loss += tf.
#     batch_loss = (loss / len(x))
#     variables = blocker.trainable_variables
#     gradients = tape.gradient(loss, variables)
#     optimizer.apply_gradients(zip(gradients, variables))
    


if __name__ == "__main__":

    X_true = np.load("./dataset/X_data.npy")

    print("data_loaded!!")

    X_fake = np.load("./dataset/fake_X_data.npy")

    X_train, _ =  train_test_split(X_true, test_size=0.4, random_state=428)


    blocker = Blocker()

    optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)

    text_length = 512

    pretrained_source = u"./Japanese_L-12_H-768_A-12_E-30_BPE_WWM"

    # tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_source)
    # embedder = BertModel.from_pretrained(pretrained_source)

    # text_ls = pd.read_csv("./Japanese-Fakenews-Dataset/trainable_fake_news.csv")["0"].to_list()

    # X_fake = blocker.preprocessing(text_ls,tokenizer,embedder)

    # np.save("./dataset/fake_X_data.npy",X_fake)
