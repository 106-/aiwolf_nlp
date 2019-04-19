#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 対話形式で分類を行う
# 使い方
# ./predict.py (辞書ファイルへのパス) (学習モデルファイルへのパス *.h5のやつ)

import MeCab
import numpy as np
from gensim.corpora import Dictionary
from gensim import matutils
from keras.models import load_model
import sys
from train_settings import CLASS_NAMES

if len(sys.argv) != 3:
    print("wrong arguments.")
    print("Usage: %s [dictfile] [modelfile]"%sys.argv[0])
    sys.exit(1)

dict_file = sys.argv[1]
model_file = sys.argv[2]

mecab = MeCab.Tagger("-Owakati")
wakati = lambda x: mecab.parse(x).split(" ")[:-1]
dct = Dictionary.load(dict_file)
types_model = load_model(model_file)

types = CLASS_NAMES

while True:
    input_str = input("-> ")
    input_wakati = wakati(input_str)
    vect_bow = matutils.corpus2dense( [dct.doc2bow(input_wakati)], num_terms=len(dct)).T

    predict_result = types_model.predict(vect_bow)[0]
    predict_class = np.argmax(np.array(predict_result))
    # 確率のリストを表示
    # print(list(zip(types, predict_result)))
    print("%s に分類!"%types[predict_class])