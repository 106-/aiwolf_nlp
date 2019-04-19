#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 学習を行う
# 使い方
# $ ./train.py (学習データへのパス)

import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
import sys
import os
from train_settings import CLASS_NUM

if len(sys.argv)<=2:
    print("wrong arguments.")
    print("Usage: %s [train_data] [layer_num layer_num...]"%sys.argv[0])
    sys.exit()

csvfile = sys.argv[1]
filedir = os.path.dirname(csvfile)
# 拡張子を除いたファイル名
filename = os.path.splitext(os.path.basename(csvfile))[0]
layers = list(map(int, sys.argv[2:]))

class LearningData:
    def __init__(self, label, data):
        self.label = label
        self.data = data

    # LearningDataをだいたい等しく分割する関数    
    def split(self, split_num=10):
        datas = np.array_split(self.data, split_num)
        labels = np.array_split(self.label, split_num)
        return list(map( lambda x: LearningData(x[0], x[1]), zip(labels, datas) ))

    @staticmethod
    def from_csv(filename, label_num):
        # csvファイルの読み込み
        table = np.loadtxt(filename, delimiter=",")
        # ランダムに並べ替える
        np.random.shuffle(table)
        # 入力データと正解データの分割
        label, data = np.hsplit(table, [1])
        # 正解データを1-of-K表現ベクトルにする
        label = to_categorical(label, label_num)
        # 入力を[0, 1]に収める(正規化という)
        data = data / np.max(data)
        return LearningData(label, data)

ld = LearningData.from_csv(csvfile, CLASS_NUM)

def fit(train, test, layers, filename, class_num, fit_setting):
    model = Sequential()
    for i, l in enumerate(layers):
        if i==0:
            model.add(Dense(l, activation="relu", input_dim=train.data.shape[1]))
        else:
            model.add(Dense(l, activation="relu"))
    model.add(Dense(class_num, activation="softmax"))
    # model.summary()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    fit = model.fit(train.data, train.label, validation_data=(test.data, test.label), **fit_setting)
    model.save(filename)
    return fit

cp = ModelCheckpoint("%s_model_chk.h5"%filename, monitor="val_loss", verbose=1,
    save_best_only=True, save_weights_only=False)

kf = KFold(n_splits=10)

def plot_history(train, test, axis, keys, title):
    axis.plot(train, label="train")
    axis.plot(test, label="validation")
    axis.set_title('model %s'%title)
    axis.set_xlabel('epoch')
    axis.set_ylabel(title)
    axis.legend()

i = 0
fig, axes = plt.subplots(ncols=2, figsize=(10,4))
acc = []
val_acc = []
loss = []
val_loss = []

for train_index, test_index in kf.split(ld.data):
    train = LearningData(ld.label[train_index], ld.data[train_index])
    test = LearningData(ld.label[test_index], ld.data[test_index])

    fit_log = fit(train, test, layers, os.path.join(filedir, "%s_model.h5"%filename), CLASS_NUM, {"batch_size":100, "epochs":100, "verbose":0, "callbacks":[]})
    
    acc.append(fit_log.history["acc"])
    val_acc.append(fit_log.history["val_acc"])
    loss.append(fit_log.history["loss"])
    val_loss.append(fit_log.history["val_loss"])

layers.insert(0, 2106)
layers.append(21)

get_mean = lambda x: np.array(x).mean(axis=0)
acc = get_mean(acc)
val_acc = get_mean(val_acc)
loss = get_mean(loss)
val_loss = get_mean(val_loss)

plot_history( acc, val_acc, axes[0], ["acc", "val_acc"], "accuracy")
plot_history( loss, val_loss, axes[1], ["loss", "val_loss"], "loss")

graph_file = "-".join(map(str, layers)) + ".eps"
fig.savefig(graph_file)
print("max acc: %g"% np.max(acc) )
print("max val_acc: %g"% np.max(val_acc) )
print("min loss: %g"% np.min(loss) )
print("min val_loss: %g"% np.min(val_loss[-1]) )