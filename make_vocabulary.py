#!/usr/bin/env python
# -*- coding:utf-8 -*-

# bowベクトルを作るための語彙を蓄えるスクリプト
# 人狼ログの中に含まれる単語を分かち書きして辞書に追加している

# 使い方
# $ ./make_vocabulary.py (csvファイルへのパス)

# csvファイルは
# (発話者), (発話内容), (分類)
# これ以外のカラムがあるとうまく動かないはず

import MeCab
import pandas as pd
from gensim.corpora import Dictionary
import os
import os.path
import sys

if len(sys.argv) != 2:
    print("wrong arguments.")
    print("Usage: %s [csvfile]"%sys.argv[0])
    sys.exit(1)

filepath = sys.argv[1]
filedir = os.path.dirname(filepath)
# 拡張子を除いたファイル名
filename = os.path.splitext(os.path.basename(filepath))[0]

# MeCabの引数をここで指定 分かち書きだけできればいいので下のオプション
mecab = MeCab.Tagger("-Owakati")

# 辞書に含めない単語たち
words_blacklist = [
    ">>",           # チャットのアノテーション
    "some_agent",
    "\u3000",       # 全角スペースを意味している
    "。",
    "、",
]

dct = Dictionary()
# csvファイルの読み込み
df = pd.read_csv(filepath, delimiter=",", names=["talker", "words", "type"])
# 文を分かち書き -> 半角スペースで区切り -> 最後の1文字(改行コード)を消したリストを得る
wakati_df = df["words"].map(lambda x: mecab.parse(x).split(" ")[:-1])
# 辞書に追加
dct.add_documents(wakati_df)

# ブラックリストの辞書内でのidを得る
words_blacklist_id = dct.doc2idx(words_blacklist)
# 辞書から削除
dct.filter_tokens(bad_ids=words_blacklist_id)
#dct.filter_n_most_frequent(600)

# 辞書の保存
dct.save(os.path.join(filedir, ".".join([filename, "dict"])))

# 辞書の中身と単語数の表示
print(dct.token2id)
print(len(dct.token2id))