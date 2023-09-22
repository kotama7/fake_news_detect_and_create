import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import joblib
from keras import backend as K
import keras

class DeEmbedder(tf.keras.Model):
    def __init__(self):
        super(DeEmbedder, self).__init__()
        self.input_dim = 1024
        self.vocab_size = 32006
        self.dense_output = Dense(self.input_dim, activation = "relu")
        self.vocab_output = Dense(self.vocab_size, activation = "softmax")
 
    def call(self, z):
        output = self.dense_output(z)
        output = self.vocab_output(output)
        return output
    
def label_make(y):
    y = y.numpy()
    data_size = len(y)
    vocab_size = 32006
    Y_ls = np.zeros((data_size, vocab_size))
    
    for j, word_id in enumerate(y):
        Y_ls[j][int(word_id)] = 1

    return tf.constant(Y_ls)

@tf.function
def train_step(x,y):
    loss = 0
    with tf.GradientTape() as tape:
        pred_y = deembeder(x, training=True)
        entoropy = tf.keras.losses.binary_crossentropy(pred_y, y)
        loss += K.sum(entoropy)
    batch_loss = (loss / len(x))
    variables = deembeder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    # accuracyの計算用
    return batch_loss


if __name__ == "__main__":

    X = np.load("./dataset/X_data.npy").reshape(-1,768)
    Y = np.load("./dataset/Y_data.npy").reshape(-1)

    print("data_loaded!!")

    # X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.4, random_state=428)

    print("data splited")

    # deembeder = DeEmbedder()

    deembeder = load_model("./model/DeEmbedder")

    optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    BATCH_SIZE = 512
    dataset_X = tf.data.Dataset.from_tensor_slices((X))
    dataset_X = dataset_X.batch(BATCH_SIZE, drop_remainder=True)
    dataset_Y = tf.data.Dataset.from_tensor_slices((Y))
    dataset_Y = dataset_Y.batch(BATCH_SIZE, drop_remainder=True)

    steps_per_epoch = len(X) // BATCH_SIZE # 何個に分けるか

    print("Start!!")
    EPOCHS = 100
    for epoch in range(EPOCHS):
        c = 0
        for (x, y) in zip(dataset_X, dataset_Y):
            c += 1
            y = label_make(y)
            batch_loss = train_step(x, y)
            if c % 100 == 0:
                print('Epoch {} Loss {}'.format(epoch + 1, batch_loss.numpy()))

        print('Epoch {} Loss {}'.format(epoch + 1, batch_loss.numpy()))
        deembeder.save("./model/DeEmbedder")