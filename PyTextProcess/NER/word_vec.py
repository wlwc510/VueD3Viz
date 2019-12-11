from gensim.models import Word2Vec
import config
import os

# src_file = config.FLAGS.src_file
src_file1 = r'W:\Rs project data\NER-损失误差\NER-master\resource\source_split_1100.txt'

word_embedding_file = config.FLAGS.word_embedding_file
file = r'W:\Rs project data\NER\NER-master\resource\word_vec\wiki.vec'


with open(src_file1, 'r', encoding='utf-8') as fp:
    content = fp.readlines()
    for i, value in enumerate(content):
        content[i] = content[i].replace('\n', '')
        content[i] = content[i].split(' ')

if not os.path.exists(file):
    model = Word2Vec(content, size=300, min_count=0, iter=1000)
    model.save(file)

model = Word2Vec.load(file)

# a = list(model['，'])
# print(",".join('%s' %id for id in a))

temp = []
for sent in content:
    temp.extend(sent)
word_set = list(set(temp))
print(len(word_set))
# print('' in word_set)

if not os.path.exists(word_embedding_file):
    with open(word_embedding_file, 'a', encoding='utf-8') as fp:
        for i, value in enumerate(word_set):
            fp.write(value + ':' + ",".join('%s' %id for id in list(model[value])) + '\n')