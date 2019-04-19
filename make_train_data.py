#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 分類に必要なデータを作成するスクリプト
# make_vocabulary.pyから得た辞書が必要

# 使い方
# $ ./make_train_data.py (辞書ファイルへのパス) (csvファイルへのパス)

import MeCab
import pandas as pd
from gensim.corpora import Dictionary
from gensim import matutils
import sys
import os.path
from train_settings import CLASS_NAMES_DICT

if len(sys.argv) != 3:
    print("wrong arguments.")
    print("Usage: %s [dictfile] [csvfile]"%sys.argv[0])
    sys.exit(1)

dictfile = sys.argv[1]
csvfile = sys.argv[2]
filedir = os.path.dirname(csvfile)
# 拡張子を除いたファイル名
filename = os.path.splitext(os.path.basename(csvfile))[0]

mecab = MeCab.Tagger("-Owakati")

# 辞書ファイルの読み込み
dct = Dictionary.load(dictfile)
# csvファイルの読み込み
df = pd.read_csv(csvfile , names=["talker", "words", "type"])

# ここから入力データの作成
# 分かち書きを行う関数を定義
wakati = lambda x: mecab.parse(x).split(" ")[:-1]
# 全ての文章を分かち書き -> bag of words表現に変換
wakati_all_df = df["words"].map(wakati).map(dct.doc2bow)
vect_bow = matutils.corpus2dense(wakati_all_df, num_terms=len(dct)).T # 最後の.Tは転置を表している

# ここから教師データの作成
# それぞれのtypeに数字を割り当てる
types = df["type"].map(CLASS_NAMES_DICT)

# 入力データと教師データの結合
# 一番最初の列成分に正解データを結合することにする.

# typeの種類データとbowなベクトルデータの結合
# vect_bowはnumpyというライブラリのデータ型で入っているので, pd.DataFrameに入れて変換している.
type_data = pd.concat([types, pd.DataFrame(vect_bow)], axis=1).astype("int64")

# 行をランダムに入れ替え
type_data = type_data.sample(frac=1)

# ファイルに出力
type_data.to_csv(os.path.join(filedir, "%s_train.csv"%filename), header=None, index=None)

# データの行数×列数を表示
print(type_data.shape)