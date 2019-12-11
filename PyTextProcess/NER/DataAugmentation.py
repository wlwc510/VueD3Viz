'''
对于深度学习方法，一般需要大量标注语料，否则极易出现过拟合，无法达到预期的泛化能力。我们在实验中发现，
通过数据增强可以明显提升模型性能。具体地，我们对原语料进行分句，然后随机地对各个句子进行拼接，
最后与原始句子一起作为训练语料。

另外，利用收集到的命名实体词典，采用随机替换的方式，用其替换语料中同类型的实体，得到增强语料。
'''

import numpy as np
import config
import os
from nltk.stem import WordNetLemmatizer
# import spacy
from collections import Counter
import re
import copy
import linecache

# 数据增强
ner_source_split_file = './resource/source.txt'
ner_target_file = './resource/target.txt'
src_file = config.FLAGS.src_file
tgt_file = config.FLAGS.tgt_file
dict_rs = config.FLAGS.dict_rs
dict_act = config.FLAGS.dict_act
dict_apl = config.FLAGS.dict_apl
dict_rs_tag = config.FLAGS.dict_rs_tag
dict_act_tag = config.FLAGS.dict_act_tag
dict_apl_tag = config.FLAGS.dict_apl_tag


def read_file(file_path):
    if not os.path.exists(file_path):
        pass
    with open(file_path, 'r', encoding='utf-8') as fp:
        content = fp.readlines()
    return content


class Preprocessor():
    def __init__(self):
        pass

    def action(self, content):
        # 扩充action词典（大小换成小写，词形还原）
        self.content = content
        wnl = WordNetLemmatizer()
        content_1 = self.content  # 后面会增加小写

        for cont in self.content:
            if cont[0].lower() != cont[0] and (cont[0].lower() + cont[1:]) not in self.content:
                content_1.append(cont[0].lower() + cont[1:])
        content_2 = content_1  # 词形还原
        for cont in content_1:
            w = wnl.lemmatize(cont, 'v')
            if w not in content_2:
                content_2.append(w)

        return content_2

    def rs(self, content):
        # 提取出remote_sensing中 括号内的简写部分
        self.content = content
        content_1 = self.content
        for sentence in self.content:
            # counter = Counter(sentence)
            ws = re.findall(r'[(](.*?)[)]', sentence) # 判断每一行是否有简写 都是在括号内
            if ws:
                for w in ws:
                    content_1.append(w[1: -1])
        return content_1

    def create_tag(self, content, B_tag_name, I_tag_name=None):
        self.content = content
        self.B_tag_name = B_tag_name
        self.I_tag_name = I_tag_name

        tag_list = []
        for sentence in self.content:
            sentence = sentence.split()
            length = len(sentence)
            if length > 1:
                tag_list.append((self.B_tag_name + (' ' + I_tag_name) * (length - 1)))
            else:
                tag_list.append(self.B_tag_name)
        return tag_list

    def sort_length(self, content, content_tag):
        # 对词典排序 从长到断
        self.content = content
        self.content_tag = content_tag

        output_dict = list(map(lambda x: x.replace('\n', ''), self.content))
        output_dict = list(sorted(output_dict, key=lambda x: -len(x.split())))
        output_dict_tag = list(map(lambda x: x.replace('\n', ''), self.content_tag))
        output_dict_tag = list(sorted(output_dict_tag, key=lambda x: -len(x)))

        return output_dict, output_dict_tag

    def creat_dict(self, target_file, source_split_file,  B_tag_name, I_tag_name=None):
        # 自动抽取命名实体，并返回列表
        '''
        :param target_file:  原标签文件
        :param source_split_file: 原文本文件（都分割开了）
        :param B_tag_name:
        :param I_tag_name:
        :return:
        '''
        self.target_file = target_file
        self.source_split_file = source_split_file
        self.B_tag_name = B_tag_name
        self.I_tag_name = I_tag_name
        with open(self.target_file, encoding='utf-8') as f:
            content = f.readlines()
            ner_word_list = []
            ner_tag_list = []
            for i, sentence in enumerate(content):
                sentence = sentence.split()
                tag_list = [] # 用于暂存标签
                for t, tag in enumerate(sentence):
                    if tag == self.B_tag_name:
                        t1 = t
                        tag_list.append(tag)
                    if tag == self.I_tag_name:
                        tag_list.append(tag)
                    if tag != self.B_tag_name and tag != self.I_tag_name and tag_list != []:
                        t2 = t
                        with open(self.source_split_file, encoding='utf-8') as fp:
                            sent = linecache.getline(self.source_split_file, i + 1).split()
                            ner_word = sent[t1: t2]
                        if ' '.join(ner_word) not in ner_word_list:
                            ner_word_list.append(' '.join(ner_word))
                            ner_tag_list.append(' '.join(tag_list))
                        tag_list = []
                        t1 = t2 = 0

        return ner_word_list, ner_tag_list

    def divide_sent(self, content):
        '''
        没写完
        切分句子，为biagrams，trigrams拼接
        :param content: 字符串 是句子
        :return: 列表 按照逗号切分
        '''
        self.content = content

        output_list = []
        for sentence in self.content:
            ws = re.findall(r'\(.*?\)', sentence)
            for w in ws:
                sentence = sentence.replace(w, 'zzffvvv')
            sent_split = sentence.split(' , ')
            b = list(map(lambda x: x + ' ,', sent_split[0: -1]))
            b.append(sent_split[-1])
            count = 0
            for i, seg in enumerate(b):
                # print(sentence)
                if 'zzffvvv' in seg:
                    b[i] = b[i].replace('zzffvvv', ws[count])
                    count += 1

            output_list.append(b)
        return output_list

    def tag_index(self, content, B_tag, I_tag=None):
        # 计算标签的索引值
        '''

        :param content: list
        :param B_tag:
        :param I_tag:
        :return: 返回标签的索引范围 list
        '''
        self.content = content
        self.B_tag = B_tag
        self.I_tag = I_tag

        # self.content = self.content.split()
        t1 = t2 = 0
        index_list = []
        for i, word in enumerate(self.content):
            if word == self.B_tag:
                t1 = i
            if self.content[i] not in [self.B_tag, self.I_tag] and self.content[i - 1] in [self.B_tag, self.I_tag]:
                t2 = i
                index_list.append([t1, t2])
                t1 = t2 = 0
        return index_list

    def replace_word(self, content, content_tag, rs_dict, rs_dict_tag, act_dict, act_dict_tag, apl_dict, apl_dict_tag):
        '''
        随机替换词典中的相同实体
        :param content:
        :param content_tag:
        :param rs_dict:
        :param act_dict:
        :param apl_dict:
        :return:
        '''
        self.content = content
        self.content_tag = content_tag
        self.rs_dict = rs_dict
        self.rs_dict_tag = rs_dict_tag
        self.act_dict = act_dict
        self.act_dict_tag = act_dict_tag
        self.apl_dict = apl_dict
        self.apl_dict_tag = apl_dict_tag

        da_sent = copy.deepcopy(self.content) #增强的数据集
        da_sent_tag = copy.deepcopy(self.content_tag)
        np.random.seed(233)
        rs_length = len(self.rs_dict)
        act_length = len(self.act_dict)
        apl_length = len(self.apl_dict)
        for (i, sentence), (t, sentence_tag) in zip(enumerate(self.content), enumerate(self.content_tag)):
            # print(i)
            sentence = sentence.split()
            sentence_tag = sentence_tag.split()
            rs_index = self.tag_index(sentence_tag, 'B-RS', 'I-RS')
            act_index = self.tag_index(sentence_tag, 'B-ACT')
            apl_index = self.tag_index(sentence_tag, 'B-APL', 'I-APL')
            all_index = rs_index + act_index + apl_index
            all_index = sorted(all_index, key=lambda x: x[0])
            if all_index != []:
                sentence_copy = sentence
                sentence_copy_tag = sentence_tag
                for i in range(10): # 重复10次
                    sentence = copy.deepcopy(sentence_copy)
                    sentence_tag = copy.deepcopy(sentence_copy_tag)
                    for single_index in all_index[: : -1]: # 反向遍历
                        if single_index in rs_index:
                            number = np.random.randint(0, rs_length - 1, 1)[0]
                            sentence[single_index[0]: single_index[1]] = self.rs_dict[number].split()
                            sentence_tag[single_index[0]: single_index[1]] = self.rs_dict_tag[number].split()
                        elif single_index in act_index:
                            number = np.random.randint(0, act_length - 1, 1)[0]
                            sentence[single_index[0]: single_index[1]] = self.act_dict[number].split()
                            sentence_tag[single_index[0]: single_index[1]] = self.act_dict_tag[number].split()
                        elif single_index in apl_index:
                            number = np.random.randint(0, apl_length - 1, 1)[0]
                            sentence[single_index[0]: single_index[1]] = self.apl_dict[number].split()
                            sentence_tag[single_index[0]: single_index[1]] = self.apl_dict_tag[number].split()
                    da_sent.append(' '.join(sentence))
                    da_sent_tag.append(' '.join(sentence_tag))
        return da_sent, da_sent_tag

    def data_aug(self, content, content_tag, rs_dict, rs_dict_tag, act_dict, act_dict_tag, apl_dict, apl_dict_tag, cycles=None):
        '''
        字典组合后与原句子相连
        :param content:
        :param content_tag:
        :param rs_dict:
        :param rs_dict_tag:
        :param act_dict:
        :param act_dict_tag:
        :param apl_dict:
        :param apl_dict_tag:
        :return:
        '''
        self.content = content
        self.content_tag = content_tag
        self.rs_dict = rs_dict
        self.rs_dict_tag = rs_dict_tag
        self.act_dict = act_dict
        self.act_dict_tag = act_dict_tag
        self.apl_dict = apl_dict
        self.apl_dict_tag = apl_dict_tag

        da_sent = []  # 增强的数据集
        da_sent_tag = []
        np.random.seed(233)
        for (i, sentence), (t, sentence_tag) in zip(enumerate(self.content), enumerate(self.content_tag)):
            sentence = sentence.split()
            sentence_tag = sentence_tag.split()
            if cycles != None:
                for c in range(cycles):
                    rs_number = np.random.randint(0, len(self.rs_dict) - 1, 1)[0]
                    act_number = np.random.randint(0, len(self.act_dict) - 1, 1)[0]
                    apl_number = np.random.randint(0, len(self.apl_dict) - 1, 1)[0]

                    sent = ', ' + self.rs_dict[rs_number].replace('\n', '') + ' ' + \
                           self.act_dict[act_number].replace('\n', '') + ' ' + \
                           self.apl_dict[apl_number].replace('\n', ' ')
                    sent_tag = 'O ' + self.rs_dict_tag[rs_number].replace('\n', '') + ' ' + \
                               self.act_dict_tag[act_number].replace('\n', '') + ' ' + \
                               self.apl_dict_tag[apl_number].replace('\n', ' ')

                    sent = sent.split()
                    sent_tag = sent_tag.split()

                    sentences = sentence[0: -1] + sent + list(sentence[-1])
                    sentences_tag = sentence_tag[0: -1] + sent_tag + list(sentence_tag[-1])
                    da_sent.append(' '.join(sentences))
                    da_sent_tag.append(' '.join(sentences_tag))
            else:
                rs_number = np.random.randint(0, len(self.rs_dict) - 1, 1)[0]
                act_number = np.random.randint(0, len(self.act_dict) - 1, 1)[0]
                apl_number = np.random.randint(0, len(self.apl_dict) - 1, 1)[0]

                sent = ', ' + self.rs_dict[rs_number].replace('\n', '') + ' ' \
                       + self.act_dict[act_number].replace('\n', '') + ' ' + \
                       self.apl_dict[apl_number].replace('\n', ' ')
                sent_tag = 'O ' + self.rs_dict_tag[rs_number].replace('\n', '') + ' ' \
                           + self.act_dict_tag[act_number].replace('\n', '') + ' ' + \
                           self.apl_dict_tag[apl_number].replace('\n', ' ')

                sent = sent.split()
                sent_tag = sent_tag.split()

                sentence = sentence[0: -1] + sent + list(sentence[-1])
                sentence_tag = sentence_tag[0: -1] + sent_tag + list(sentence_tag[-1])
                da_sent.append(' '.join(sentence))
                da_sent_tag.append(' '.join(sentence_tag))
        return da_sent, da_sent_tag


pre = Preprocessor()
content = read_file(ner_source_split_file)
content = list(map(lambda x: x.replace('\n', ''), content))
# print(content[0])
print(len(content))

content_tag = read_file(ner_target_file)
content_tag = list(map(lambda x: x.replace('\n', ''), content_tag))
# print(content_tag[0])
print(len(content_tag))


dict_rs = read_file(dict_rs)
dict_rs = list(map(lambda x: x.replace('\n', ''), dict_rs))
dict_act = read_file(dict_act)
dict_act = list(map(lambda x: x.replace('\n', ''), dict_act))
dict_apl = read_file(dict_apl)
dict_apl = list(map(lambda x: x.replace('\n', ''), dict_apl))
dict_rs_tag = read_file(dict_rs_tag)
dict_rs_tag = list(map(lambda x: x.replace('\n', ''), dict_rs_tag))
dict_act_tag = read_file(dict_act_tag)
dict_act_tag = list(map(lambda x: x.replace('\n', ''), dict_act_tag))
dict_apl_tag = read_file(dict_apl_tag)
dict_apl_tag = list(map(lambda x: x.replace('\n', ''), dict_apl_tag))

# da, da_tag = pre.replace_word(content, content_tag, dict_rs, dict_rs_tag, dict_act, dict_act_tag, dict_apl, dict_apl_tag)
da, da_tag = pre.data_aug(content, content_tag, dict_rs, dict_rs_tag, dict_act, dict_act_tag, dict_apl, dict_apl_tag, 10)

print(len(da), len(da_tag))

# print(da[300: 200])
for i in range(len(da)):
    if (len(da[i].split()) != len(da_tag[i].split())):
        print(i)
        print(da[i].split())
        print(da_tag[i].split())



with open(src_file, 'a', encoding='utf-8') as f1:
    if f1.readlines() != '':
        for i in da:
            f1.write(i + '\n')

with open(tgt_file, 'a', encoding='utf-8') as f2:
    if f2.readlines() != '':
        for i in da_tag:
            f2.write(i + '\n')

