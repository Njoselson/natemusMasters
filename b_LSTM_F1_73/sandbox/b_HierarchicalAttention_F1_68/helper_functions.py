import emoji
import re
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from keras import backend as K
from keras import optimizers
from keras.models import Model
import re
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatibl|e with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def is_emoji(s):
    return s in emoji.UNICODE_EMOJI

def remove_emoji(text):
    letter_array = [char for char in text if not is_emoji(char)]
    return ''.join(letter_array)

def add_space(text):
    return ''.join(' ' + char + ' ' if is_emoji(char) else char for char in text).strip()

def remove_text(text):
    words = text.split(' ')
    emojis = [word for word in words if is_emoji(word)]
    return emojis

def count_length(text):
    return len(text)

def count_upper_case(text):
    return sum(1 for c in text if c.isupper())

def find_converage(big_dict, small_dict):
    not_in_small = {}
    missing = 0
    exist_both = 0
    for key in small_dict:
        if key in big_dict:
            exist_both += 1
        else:
            not_in_small[key] = 0
            missing += 1
    coverage = exist_both/(exist_both + missing)
    return [coverage, missing, exist_both], not_in_small

def add_word(string, word):
    return 1 if word in string.split(' ') else 0

def find_number_missing(not_in_small, tokenizer_input_data):
    loop_count = 0
    for line in tokenizer_input_data:
        for key, _ in not_in_small.items():
            not_in_small[key] += add_word(line, key)
        loop_count += 1
        print("%s of %s"%(loop_count, len(tokenizer_input_data)))
    return not_in_small

def fix_apos(word):
    replaced_word = re.sub(r"(it['|´|’]s)", "it is", word)
    replaced_word = re.sub(r"(can['|´|’]t)", "can not", replaced_word)
    replaced_word = re.sub(r"(ain['|´|’]t)", "am not", replaced_word)
    replaced_word = re.sub(r"([I|a-z]{0,10})['|´|’]re", r"\1 are", replaced_word)
    replaced_word = re.sub(r"([I|a-z]{0,10})['|´|’]ll", r"\1 will", replaced_word)
    replaced_word = re.sub(r"([I|a-z]{0,10})['|´|’]ve", r"\1 have", replaced_word)
    replaced_word = re.sub(r"(he|who|how|when|there)['|´|’]s", r"\1 is", replaced_word)
    replaced_word = re.sub(r"that['|´|’]s", r"that is", replaced_word)
    replaced_word = re.sub(r"what['|´|’]s", r"what is", replaced_word)
    replaced_word = re.sub(r"(let['|´|’]s)", "let us", replaced_word)
    replaced_word = re.sub(r"([I|i]['|´|’]m)", "i am", replaced_word)
    replaced_word = re.sub(r"(won['|´|’]t)", "will not", replaced_word)
    replaced_word = re.sub(r"(n['|´|’]t)", " not", replaced_word)
    replaced_word = re.sub(r"['|´|’]", r" ", replaced_word)
    return replaced_word

def str2emoji(text):
    text = re.sub(r":‑\)"," ☺️ ",text)
    text = re.sub(r":\)\)\)\)"," ☺️ ",text)
    text = re.sub(r":\)\)\)"," ☺️ ",text)
    text = re.sub(r":\)\)"," ☺️ ",text)
    text = re.sub(r":\)"," ☺️ ",text)
    text = re.sub(r":-\]"," ☺️ ",text)
    text = re.sub(r":\]"," ☺️ ",text)
    text = re.sub(r":-3"," ☺️ ",text)
    text = re.sub(r":3","  ☺️ ",text)
    text = re.sub(r":->"," ☺️ ",text)
    text = re.sub(r"</3",' 💔 ',text)
    text = re.sub(r"<3"," ❤️ ",text)
    text = re.sub(r":>"," ☺️ ",text)
    text = re.sub(r"8-\)"," ☺️ ",text)
    text = re.sub(r":o\)"," ☺️ ",text)
    text = re.sub(r":-\}"," ☺️ ",text)
    text = re.sub(r":\}"," ☺️ ",text)
    text = re.sub(r":-\)"," ☺️ ",text)
    text = re.sub(r":c\)"," ☺️ ",text)
    text = re.sub(r":\^\)"," ☺️ ",text)
    text = re.sub(r"=\]"," ☺️ ",text)
    text = re.sub(r"=\)"," ☺️ ",text)
    text = re.sub(r":‑D"," 😃 ",text)
    text = re.sub(r":D"," 😃 ",text)
    text = re.sub(r"8‑D"," 😃 ",text)
    text = re.sub(r"8D"," 😃 ",text)
    text = re.sub(r"X‑D"," 😃 ",text)
    text = re.sub(r"XD"," 😃 ",text)
    text = re.sub(r"=D"," 😃 ",text)
    text = re.sub(r"=3"," 😃 ",text)
    text = re.sub(r"B\^D"," 😃 ",text)
    text = re.sub(r":-\)\)"," 😃 ",text)
    text = re.sub(r":‑\("," ☹️ ",text)
    text = re.sub(r":-\("," ☹️ ",text)
    text = re.sub(r":-d", " 🤤 ",text)
    text = re.sub(r":d", " 🤤 ",text)
    text = re.sub(r"=d", " 🤤 ",text)

    text = re.sub(r":\("," ☹️ ",text)
    text = re.sub(r":‑c"," ☹️ ",text)
    text = re.sub(r":c"," ☹️ ",text)
    text = re.sub(r":‑<"," ☹️ ",text)
    text = re.sub(r":<"," ☹️ ",text)
    text = re.sub(r":‑\["," ☹️ ",text)
    text = re.sub(r":\["," ☹️ ",text)
    text = re.sub(r":-\|\|"," ☹️ ",text)
    text = re.sub(r">:\["," ☹️ ",text)
    text = re.sub(r":\{"," ☹️ ",text)
    text = re.sub(r":@"," ☹️ ",text)
    text = re.sub(r">:\("," ☹️ ",text)
    text = re.sub(r":'‑\("," 😭 ",text)
    text = re.sub(r":'\("," 😭 ",text)
    text = re.sub(r":'‑\)"," 😃 ",text)
    text = re.sub(r":'\)"," 😃 ",text)
    text = re.sub(r"D‑':"," 😧 ",text)
    text = re.sub(r"D:<"," 😨 ",text)
    text = re.sub(r"D:"," 😧 ",text)
    text = re.sub(r"D8"," 😧 ",text)
    text = re.sub(r"D;"," 😧 ",text)
    text = re.sub(r"D="," 😧 ",text)
    text = re.sub(r"DX"," 😧 ",text)
    text = re.sub(r":‑O"," 😮 ",text)
    text = re.sub(r":O"," 😮 ",text)
    text = re.sub(r":‑o"," 😮 ",text)
    text = re.sub(r":o"," 😮 ",text)
    text = re.sub(r":-0"," 😮 ",text)
    text = re.sub(r"8‑0"," 😮 ",text)
    text = re.sub(r">:O"," 😮 ",text)
    text = re.sub(r":-\*"," 😗 ",text)
    text = re.sub(r":\*"," 😗 ",text)
    text = re.sub(r":X"," 😗 ",text)
    text = re.sub(r";‑\)"," 😉 ",text)
    text = re.sub(r";\)"," 😉 ",text)
    text = re.sub(r"\*-\)"," 😉 ",text)
    text = re.sub(r"\*\)"," 😉 ",text)
    text = re.sub(r";‑\]"," 😉 ",text)
    text = re.sub(r";\]"," 😉 ",text)
    text = re.sub(r";\^\)"," 😉 ",text)
    text = re.sub(r":‑,"," 😉 ",text)
    text = re.sub(r";D"," 😉 ",text)
    text = re.sub(r":‑P"," 😛 ",text)
    text = re.sub(r":‑p"," 😛 ",text)
    text = re.sub(r":P"," 😛 ",text)
    text = re.sub(r":p"," 😛 ",text)
    text = re.sub(r"X‑P"," 😛 ",text)
    text = re.sub(r":‑Þ"," 😛 ",text)
    text = re.sub(r":Þ"," 😛 ",text)
    text = re.sub(r":b"," 😛 ",text)
    text = re.sub(r"d:"," 😛 ",text)
    text = re.sub(r"=p"," 😛 ",text)
    text = re.sub(r">:P"," 😛 ",text)
    text = re.sub(r":‑/"," 😕 ",text)
    text = re.sub(r":/"," 😕 ",text)
    text = re.sub(r":-\[\.\]"," 😕 ",text)
    text = re.sub(r">:/"," 😕 ",text)
    text = re.sub(r"=/"," 😕 ",text)
    text = re.sub(r":L"," 😕 ",text)
    text = re.sub(r"=L"," 😕 ",text)
    text = re.sub(r":S"," 😕 ",text)
    text = re.sub(r":‑\|"," 😐 ",text)
    text = re.sub(r":\|"," 😐 ",text)
    text = re.sub(r":$"," 😳 ",text)
    text = re.sub(r":‑x"," 🤐 ",text)
    text = re.sub(r":x"," 🤐 ",text)
    text = re.sub(r":‑#"," 🤐 ",text)
    text = re.sub(r":#"," 🤐 ",text)
    text = re.sub(r":‑&"," 🤐 ",text)
    text = re.sub(r":&"," 🤐 ",text)
    text = re.sub(r"O:‑\)"," 😇 ",text)
    text = re.sub(r"O:\)"," 😇 ",text)
    text = re.sub(r"0:‑3"," 😇 ",text)
    text = re.sub(r"0:3"," 😇 ",text)
    text = re.sub(r"0:‑\)"," 😇 ",text)
    text = re.sub(r"0:\)"," 😇 ",text)
    text = re.sub(r":‑b"," 😛 ",text)
    text = re.sub(r"0;\^\)"," 😇 ",text)
    text = re.sub(r">:‑\)"," 😈 ",text)
    text = re.sub(r">:\)"," 😈 ",text)
    text = re.sub(r"\}:‑\)"," 😈 ",text)
    text = re.sub(r"\}:\)"," 😈 ",text)
    text = re.sub(r"3:‑\)"," 😈 ",text)
    text = re.sub(r"3:\)"," 😈 ",text)
    text = re.sub(r">;\)"," 😈 ",text)
    text = re.sub(r"\|;‑\)"," 😎 ",text)
    text = re.sub(r"\|‑O"," 😏 ",text)
    text = re.sub(r":‑J"," 😏 ",text)
    text = re.sub(r"%‑\)"," 😵 ",text)
    text = re.sub(r"%\)"," 😵 ",text)
    text = re.sub(r":-###.."," 🤒 ",text)
    text = re.sub(r":###.."," 🤒 ",text)
    text = re.sub(r"\(>_<\)"," 😣 ",text)
    text = re.sub(r"\(>_<\)>"," 😣 ",text)
    text = re.sub(r"\(';'\)"," 👶 ",text)
    text = re.sub(r"\(\^\^>``"," 😓 ",text)
    text = re.sub(r"\(\^_\^;\)"," 😓 ",text)
    text = re.sub(r"\(-_-;\)"," 😓 ",text)
    text = re.sub(r"\(~_~;\) \(・\.・;\)"," 😓 ",text)
    text = re.sub(r"\(-_-\)zzz"," 😴 ",text)
    text = re.sub(r"\(\^_-\)"," 😉 ",text)
    text = re.sub(r"\(\(\+_\+\)\)"," 😕 ",text)
    text = re.sub(r"\(\+o\+\)"," 😕 ",text)
    text = re.sub(r"\^_\^"," 😃 ",text)
    text = re.sub(r"\(\^_\^\)/"," 😃 ",text)
    text = re.sub(r"\(\^O\^\)／"," 😃 ",text)
    text = re.sub(r"\(\^o\^\)／"," 😃 ",text)
    text = re.sub(r"\(__\)"," 🙇 ",text)
    text = re.sub(r"_\(\._\.\)_"," 🙇 ",text)
    text = re.sub(r"<\(_ _\)>"," 🙇 ",text)
    text = re.sub(r"<m\(__\)m>"," 🙇 ",text)
    text = re.sub(r"m\(__\)m"," 🙇 ",text)
    text = re.sub(r"m\(_ _\)m"," 🙇 ",text)
    text = re.sub(r"\('_'\)"," 😭 ",text)
    text = re.sub(r"\(/_;\)"," 😭 ",text)
    text = re.sub(r"\(T_T\) \(;_;\)"," 😭 ",text)
    text = re.sub(r"\(;_;"," 😭 ",text)
    text = re.sub(r"\(;_:\)"," 😭 ",text)
    text = re.sub(r"\(;O;\)"," 😭 ",text)
    text = re.sub(r"\(:_;\)"," 😭 ",text)
    text = re.sub(r"\(ToT\)"," 😭 ",text)
    text = re.sub(r";_;"," 😭 ",text)
    text = re.sub(r";-;"," 😭 ",text)
    text = re.sub(r";n;"," 😭 ",text)
    text = re.sub(r";;"," 😭 ",text)
    text = re.sub(r"Q\.Q"," 😭 ",text)
    text = re.sub(r"T\.T"," 😭 ",text)
    text = re.sub(r"QQ"," 😭 ",text)
    text = re.sub(r"Q_Q"," 😭 ",text)
    text = re.sub(r"\(-\.-\)"," 😞 ",text)
    text = re.sub(r"\(-_-\)"," 😞 ",text)
    text = re.sub(r"-_-"," 😞 ",text)
    text = re.sub(r"\(一一\)"," 😞 ",text)
    text = re.sub(r"\(；一_一\)"," 😞 ",text)
    text = re.sub(r"\(=_=\)"," 😩 ",text)
    text = re.sub(r"\(=\^\·\^=\)"," 😺 ",text)
    text = re.sub(r"\(=\^\·\·\^=\)"," 😺 ",text)
    text = re.sub(r"=_\^= "," 😺 ",text)
    text = re.sub(r"\(\.\.\)"," 😔 ",text)
    text = re.sub(r"\(\._\.\)"," 😔 ",text)
    text = re.sub(r"\(\・\・\?"," 😕 ",text)
    text = re.sub(r"\(\?_\?\)"," 😕 ",text)
    text = re.sub(r">\^_\^<"," 😃 ",text)
    text = re.sub(r"<\^!\^>"," 😃 ",text)
    text = re.sub(r"\^/\^"," 😃 ",text)
    text = re.sub(r"\（\*\^_\^\*）"," 😃 ",text)
    text = re.sub(r"\(\^<\^\) \(\^\.\^\)"," 😃 ",text)
    text = re.sub(r"\(^\^\)"," 😃 ",text)
    text = re.sub(r"\(\^\.\^\)"," 😃 ",text)
    text = re.sub(r"\(\^_\^\.\)"," 😃 ",text)
    text = re.sub(r"\(\^_\^\)"," 😃 ",text)
    text = re.sub(r"\(\^\^\)"," 😃 ",text)
    text = re.sub(r"\(\^J\^\)"," 😃 ",text)
    text = re.sub(r"\(\*\^\.\^\*\)"," 😃 ",text)
    text = re.sub(r"\(\^—\^\）"," 😃 ",text)
    text = re.sub(r"\(#\^\.\^#\)"," 😃 ",text)
    text = re.sub(r"\（\^—\^\）"," 👋 ",text)
    text = re.sub(r"\(;_;\)/~~~"," 👋 ",text)
    text = re.sub(r"\(\^\.\^\)/~~~"," 👋 ",text)
    text = re.sub(r"\(T_T\)/~~~"," 👋 ",text)
    text = re.sub(r"\(\*\^0\^\*\)"," 😍 ",text)
    text = re.sub(r"\(\*_\*\)"," 😍 ",text)
    text = re.sub(r"\(\*_\*;"," 😍 ",text)
    text = re.sub(r"\(\+_\+\) \(@_@\)"," 😍 ",text)
    text = re.sub(r"\(\*\^\^\)v"," 😂 ",text)
    text = re.sub(r"\(\^_\^\)v"," 😂 ",text)
    text = re.sub(r"\(ーー;\)"," 😓 ",text)
    text = re.sub(r"\(\^0_0\^\)"," 😎 ",text)
    text = re.sub(r"\(\＾ｖ\＾\)"," 😀 ",text)
    text = re.sub(r"\(\＾ｕ\＾\)"," 😀 ",text)
    text = re.sub(r"\(\^\)o\(\^\)"," 😀 ",text)
    text = re.sub(r"\(\^O\^\)"," 😀 ",text)
    text = re.sub(r"\(\^o\^\)"," 😀 ",text)
    text = re.sub(r"\)\^o\^\("," 😀 ",text)
    text = re.sub(r":O o_O"," 😮 ",text)
    text = re.sub(r"o_0"," 😮 ",text)
    text = re.sub(r"o\.O"," 😮 ",text)
    text = re.sub(r"\(o\.o\)"," 😮 ",text)
    text = re.sub(r"oO"," 😮 ",text)
    text = re.sub(r':\‑\)','😃',text)
    text = re.sub(r":\)"," ☺️ ",text)
    text = re.sub(r":-]"," ☺️ ",text)
    text = re.sub(r":]"," ☺️ ",text)
    text = re.sub(r"8\-\)"," ☺️ ",text)
    text = re.sub(r":o\)"," ☺️ ",text)
    text = re.sub(r":-}"," ☺️ ",text)
    text = re.sub(r":}"," ☺️ ",text)
    text = re.sub(r":\-\)"," ☺️ ",text)
    text = re.sub(r":c\)"," ☺️ ",text)
    text = re.sub(r":^\)"," ☺️ ",text)
    text = re.sub(r"=]"," ☺️ ",text)
    text = re.sub(r"=\)"," ☺️ ",text)
    text = re.sub(r"B^D"," 😃 ",text)
    text = re.sub(r":-\)\)"," 😃 ",text)
    text = re.sub(r":-\("," ☹️ ",text)

    text = re.sub(r":‑\("," ☹️ ",text)
    text = re.sub(r":\("," ☹️ ",text)
    text = re.sub(r":\‑\["," ☹️ ",text)
    text = re.sub(r":\["," ☹️ ",text)
    text = re.sub(r":-\|\|"," ☹️ ",text)
    text = re.sub(r">\:\["," ☹️ ",text)
    text = re.sub(r":{"," ☹️ ",text)
    text = re.sub(r">\:\("," ☹️ ",text)
    text = re.sub(r":'‑\("," 😭 ",text)
    text = re.sub(r":'\("," 😭 ",text)
    text = re.sub(r":'\‑\)"," 😃 ",text)
    text = re.sub(r":'\)"," 😃 ",text)
    text = re.sub(r":-\*"," 😗 ",text)
    text = re.sub(r":\*"," 😗 ",text)
    text = re.sub(r";\‑\)"," 😉 ",text)
    text = re.sub(r";\)"," 😉 ",text)
    text = re.sub(r"\*\-\)"," 😉 ",text)
    text = re.sub(r"\*\)"," 😉 ",text)
    text = re.sub(r";‑\]"," 😉 ",text)
    text = re.sub(r";\]"," 😉 ",text)
    text = re.sub(r";^\)"," 😉 ",text)
    text = re.sub(r">\:\[\(\)\]"," 😕 ",text)

    text = re.sub(r":\[\(\)\]"," 😕 ",text)
    text = re.sub(r"=\[\(\)\]"," 😕 ",text)
    text = re.sub(r":‑\|"," 😐 ",text)
    text = re.sub(r":\|"," 😐 ",text)
    text = re.sub(r"O:‑\)"," 😇 ",text)
    text = re.sub(r"O:\)"," 😇 ",text)
    text = re.sub(r"0:‑\)"," 😇 ",text)
    text = re.sub(r"0:\)"," 😇 ",text)
    text = re.sub(r"0;^\)"," 😇 ",text)
    text = re.sub(r">:‑\)"," 😈 ",text)
    text = re.sub(r">:\)"," 😈 ",text)
    text = re.sub(r"}:‑\)"," 😈 ",text)
    text = re.sub(r"}:\)"," 😈 ",text)
    text = re.sub(r"3:‑\)"," 😈 ",text)
    text = re.sub(r"3:\)"," 😈 ",text)
    text = re.sub(r">;\)"," 😈 ",text)
    text = re.sub(r"\|;‑\)"," 😎 ",text)
    text = re.sub(r"\|‑O"," 😏 ",text)
    text = re.sub(r"%‑\)"," 😵 ",text)
    text = re.sub(r"%\)"," 😵 ",text)
    text = re.sub(r"\(>_<\)"," 😣 ",text)
    text = re.sub(r"\(>_<\)>"," 😣 ",text)
    text = re.sub(r"\(';'\)"," Baby ",text)
    text = re.sub(r"\(^^>``"," 😓 ",text)
    text = re.sub(r"\(^_^;\)"," 😓 ",text)
    text = re.sub(r"\(-_-;\)"," 😓 ",text)

    text = re.sub(r"\(~_~;\) \(・\.・;\)"," 😓 ",text)
    text = re.sub(r"\(-_-\)zzz"," 😴 ",text)
    text = re.sub(r"\(^_-\)"," 😉 ",text)
    text = re.sub(r"\(\(\+_\+\)\)"," 😕 ",text)
    text = re.sub(r"\(\+o\+\)"," 😕 ",text)
    text = re.sub(r"^_^"," 😃 ",text)
    text = re.sub(r"\(^_^\)/"," 😃 ",text)
    text = re.sub(r"\(^O^\)／"," 😃 ",text)
    text = re.sub(r"\(__\)"," 🙇 ",text)
    text = re.sub(r"_\(._.\)_"," 🙇 ",text)
    text = re.sub(r"<\(_ _\)>"," 🙇 ",text)
    text = re.sub(r"<m\(__\)m>"," 🙇 ",text)
    text = re.sub(r"m\(__\)m"," 🙇 ",text)
    text = re.sub(r"m\(_ _\)m"," 🙇 ",text)
    text = re.sub(r"\('_'\)"," 😭 ",text)
    text = re.sub(r"\(/_;\)"," 😭 ",text)
    text = re.sub(r"\(T_T\) (;_;)"," 😭 ",text)
    text = re.sub(r"\(;_;"," 😭 ",text)
    text = re.sub(r"\(;_:\)"," 😭 ",text)
    text = re.sub(r"\(;O;\)"," 😭 ",text)
    text = re.sub(r"\(:_;\)"," 😭 ",text)
    text = re.sub(r"\(ToT\)","  😭  ",text)
    text = re.sub(r"Q\.Q"," 😭 ",text)
    text = re.sub(r"T\.T"," 😭 ",text)
    text = re.sub(r"\(-\.-\)"," 😞 ",text)
    text = re.sub(r"\(-_-\)"," 😞 ",text)

    text = re.sub(r"\(一一\)"," 😞 ",text)
    text = re.sub(r"\(；一_一\)"," 😞 ",text)

    text = re.sub(r"\(=\_=\)"," 😩 ",text)
    text = re.sub(r"\(=^\·^=\)"," 😺 ",text)

    text = re.sub(r"\(=^··^=\)"," 😺 ",text)
    text = re.sub(r"=_^= "," 😺 ",text)

    text = re.sub(r"\(\.\.\)"," 😔 ",text)

    text = re.sub(r"\(\._\.\)"," 😔  ",text)
    text = re.sub(r"\(・・\?"," 😕 ",text)
    text = re.sub(r"\(\?_\?\)"," 😕 ",text)

    text = re.sub(r">^_^<"," 😃 ",text)
    text = re.sub(r"<^\!^>"," 😃 ",text)
    text = re.sub(r"^/^","  😃  ",text)
    text = re.sub(r"\(\*^_^\*\)","  😃  ",text)
    text = re.sub(r"\(^^\)","  😃  ",text)
    text = re.sub(r"\(^\.^\)","  😃  ",text)
    text = re.sub(r"\(^_^\.\)","  😃  ",text)
    text = re.sub(r"\(^_^\)","  😃  ",text)
    text = re.sub(r"\(^J^\)","  😃  ",text)
    text = re.sub(r"\(\*^\.^\*\)"," 😃  ",text)
    text = re.sub(r"\(^—^\）"," 😃 ",text)

    text = re.sub(r"\(#^.^#\)"," 😃 ",text)
    text = re.sub(r"\(^—^\)"," 👋 ",text)
    text = re.sub(r"\(;_;\)/~~~","  👋  ",text)

    text = re.sub(r"\(^.^\)/~~~"," 👋 ",text)
    text = re.sub(r"\(-_-\)/~~~ \($··\)/~~~"," 👋 ",text)
    text = re.sub(r"\(T_T\)/~~~"," 👋 ",text)

    text = re.sub(r"\(ToT\)/~~~"," 👋 ",text)
    text = re.sub(r"\(\*^0^\*\)"," 😍 ",text)
    text = re.sub(r"\(\*_\*\)"," 😍 ",text)

    text = re.sub(r"\(\*_\*;"," 😍 ",text)
    text = re.sub(r"\(+_+\) \(@_@\)"," 😍 ",text)
    text = re.sub(r"o\.O"," 😮 ",text)
    text = re.sub(r"\(o\.o\)"," 😮 ",text)

    return text
