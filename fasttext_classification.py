# -*- coding: utf-8 -*-

import jieba
import csv
import re
import fasttext
# import skift
import pandas

#读取训练数据并进行分词
filename = 'data/train_first.csv'
train_data = open("data/train_data.txt",'a',encoding="utf-8")
segement_file=open("segement_file.txt",'a',encoding="utf-8")
with open(filename,"r",encoding="utf-8") as f:
    reader = csv.reader(f)
    # print(list(reader))
    for row in reader:
        if row[0]!="Id" and row[1]!="Discuss" and row[2]!="Score":
            data_id=row[0]
            data_discuss=row[1]
            data_score=row[2]
            # print("data id : ",data_id)
            # print("data discuss : ",data_discuss)
            # print("data score : ",data_score)
            data_discuss=str(data_discuss)
            data_discuss=re.sub('[\&\<\>\%\*\@\#\^\\/\\\]', '', data_discuss)
            data_discuss=re.sub(' ', '', data_discuss)
            data_discuss=re.sub('br', '', data_discuss)
            data_discuss=re.sub('rn', '', data_discuss)
            data_discuss = re.sub('、', '', data_discuss)
            data_discuss=re.sub('，', '', data_discuss)
            data_discuss = re.sub('。', '', data_discuss)
            data_discuss = re.sub(r'【[^{}]*】', '', data_discuss)
            words = jieba.cut(data_discuss)
            words=list(words)
            for word in words:
                train_data.write(word)
                train_data.write(" ")
                #print(" ".join(str(word)).encode('utf-8'))
            train_data.write("__label__" + data_score)
            train_data.write("\n")
train_data.close()

#读取测试数据并进行分词
test_filename = 'data/predict_first.csv'
test_data = open("data/test_data.txt",'a',encoding="utf-8")
with open(test_filename,"r",encoding="utf-8") as f:
    reader = csv.reader(f)
    # print(list(reader))
    for row in reader:
        if row[0]!="Id" and row[1]!="Discuss":
            data_id=row[0]
            data_discuss=row[1]
            data_discuss=str(data_discuss)
            data_discuss=re.sub('[\&\<\>\%\*\@\#\^\\/\\\]', '', data_discuss)
            data_discuss=re.sub(' ', '', data_discuss)
            data_discuss=re.sub('br', '', data_discuss)
            data_discuss=re.sub('rn', '', data_discuss)
            data_discuss = re.sub('、', '', data_discuss)
            data_discuss=re.sub('，', '', data_discuss)
            data_discuss = re.sub('。', '', data_discuss)
            data_discuss = re.sub(r'【[^{}]*】', '', data_discuss)
            words = jieba.cut(data_discuss)
            words=list(words)
            for word in words:
                test_data.write(word)
                test_data.write(" ")
                #print(" ".join(str(word)).encode('utf-8'))
            test_data.write("\n")
test_data.close()

#利用fasttext进行分类
#用自己的数据集进行训练
classifier = fasttext.supervised('data/train_data.txt','evaluation_predict_model')
print(type(classifier))
# 导入训练好的标准模型
# classifier =fasttext.load_model('news_fasttext.model.bin', label_prefix='__label__')
# 进行测试
result = classifier.test("data/test_data.txt")
print(result.precision)
print(result.recall)
# 对单个数据进行测试
texts = ['岛上 看 日落 的 地方 视野 很 开阔 非常 漂亮', '很 有 鲁迅 风味 很 喜欢 这样 有 文化 的 地方 ']
result_test = classifier.predict(texts)
print(result_test)

# 用skift进行分类
# from skift import FirstColFtClassifier
# df = pandas.DataFrame([['woof', 0], ['meow', 1]], columns=['txt', 'lbl'])
# sk_clf = FirstColFtClassifier(lr=0.3, epoch=10)
# sk_clf.fit(df[['txt']], df['lbl'])
# sk_clf.predict([['woof']])