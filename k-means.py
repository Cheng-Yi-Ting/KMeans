# coding:utf-8
__author__ = "liuxuejiang"
import jieba
import jieba.posseg as pseg
import os
import sys
import math
import json
from collections import OrderedDict

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans


class TF_IDF():
    def __init__(self):
        self.docs = {}
        self.seg_docs = self.get_seg_docs()
        self.stopword = []
        self.tf = []
        self.df = {}
        self.idf = {}
        self.topK_idf = {}
        self.bow = {}

    def read_file(self, path, type):
        # file.read([size])从文件读取指定的字节数，如果未给定或为负则读取所有。
        if type == 'json':
            with open(path, 'r', encoding='utf-8') as file:
                data = json.loads(file.read())
        elif type == 'txt':
            with open(path, 'r', encoding='utf-8') as file:
                data = file.read()
        return data

    def get_seg_docs(self):
        _seg_docs = []
        FOLDER_NAME = 'data'
        # DOCUMENT = 'test.json'
        DOCUMENT = 'ettoday.news.json'
        STOPWORD = 'stopword.txt'
        # 其中__file__虽然是所在.py文件的完整路径，但是这个变量有时候返回相对路径，有时候返回绝对路径，因此还要用os.path.realpath()函数来处理一下。
        # 获取当前文件__file__的路径，    __file__是当前执行的文件
        FILE_DIR = os.path.join(os.path.split(
            os.path.realpath(__file__))[0], FOLDER_NAME)

        self.docs = self.read_file(FILE_DIR + '/' + DOCUMENT, 'json')
        self.stopword = self.read_file(FILE_DIR + '/' + STOPWORD, 'txt')
        self.ca = []
        # jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
        # jieba.lcut 以及 jieba.lcut_for_search 直接返回 list
        # isalpha()去除不是字母組成的字，中文字也算，e.g:\r\n，不然會斷出\r\n
        for i in range(len(self.docs)):
            # 計算幾個類別
            # self.ca.append(self.docs[i]['category'])
            content_str = ''
            # content_seg = []
            for w in jieba.lcut(self.docs[i]['content']):
                if len(w) > 1 and w not in self.stopword and w.isalpha():
                    content_str = content_str+' '+w
                    # content_seg.append(w)
            _seg_docs.append(content_str)
        # print(self.ca)
        category = list(set(self.ca))
        # print(category)
        return _seg_docs


if __name__ == '__main__':
    tf_idf = TF_IDF()
    # print(tf_idf.seg_docs)
    corpus = tf_idf.seg_docs
    # corpus = ["我 来到 北京 清华大学",  # 第一类文本切词后的结果，词之间以空格隔开
    #           "他 来到 了 网易 杭研 大厦",  # 第二类文本的切词结果
    #           "小明 硕士 毕业 与 中国 科学院",  # 第三类文本的切词结果
    #           "我 爱 北京 天安门"]  # 第四类文本的切词结果
    # CountVectorizer类会将文本中的词语转换为词频矩阵，例如矩阵中包含一个元素a[i][j]，它表示j词在i类文本下的词频。
    # 它通过fit_transform函数计算各个词语出现的次数，通过get_feature_names()可获取词袋中所有文本的关键字，通过toarray()可看到词频矩阵的结果。
    # TfidfTransformer用于统计vectorizer中每个词语的TF-IDF值。
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # print(vectorizer.fit_transform(corpus))
    # print(tfidf)
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # print(word)
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    print(weight)
    # for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     print("-------输出第", i, "類文本的字詞tf-idf權重------")
    #     for j in range(len(word)):
    #         print(word[j], weight[i][j])
    print('Start Kmeans:')
    # kmeans演算法，分成12群，weight為tfidf矩陣

    clf = KMeans(n_clusters=28).fit(weight)
    # s = clf.fit(weight)
    # print(s)
    # print('--------------------------')
    # # n_clusters个中心点
    print(clf.cluster_centers_)
    # print('--------------------------')
    # # 每个样本所属的簇
    print(clf.labels_)
    # print('--------------------------')

    # 增加label欄位到json中，判斷新聞文章屬於哪個群集
    # for i in range(len(tf_idf.docs)):
    #     tf_idf.docs[i]['label'] = clf.labels_[i]
    #     # print(i, tf_idf.docs[i])
    # # 排序，相同群集合在一起
    # lines = sorted(tf_idf.docs, key=lambda k: k['label'])

    # i = 0
    # while i < len(tf_idf.docs):
    #     print(i, ':', lines[i]['title'], " ", lines[i]
    #           ['category'], " ", lines[i]['label'])
    #     i = i+1
    # for line in lines:
    #     print(i, ':', line.title, " ", line.label)
    #     i = i+1
    # i = 0
    # while i < len(clf.labels_):
    #     print(i, ':', tf_idf.docs[i]['title'],
    #           tf_idf.docs[i]['category'], clf.labels_[i])
    #     i = i + 1
    # print('--------------------------')
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    # print(clf.inertia_)
