from fake_news_blocker import Blocker
from fake_news_trainer import VAE
import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model

if not os.path.exists("./model/blocker"):
    blocker = Blocker()
else:
    blocker = load_model("./model/blocker")

if not os.path.exists("./model/vae"):
    vae = VAE()
else:
    vae = load_model("./model/vae")

