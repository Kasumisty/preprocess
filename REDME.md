# 简要介绍

## 文件说明

### processed_data

processed_data 中存放着处理过的数据：
1. combination.xml 中保存的是所有事件数据的集合（只取一个事件的第一个mention不包含/view/datacheck.txt中的文件内容）。
2. predata.txt 为将事件处理成自定义格式之后的形式。
3. predata_with_all_mentions.txt 为所有事件数据的集合（包含全部mentions且不包含/view/datacheck.txt中的文件内容）。
4. trian_data.pkl 处理过后的训练集数据（包括wordembedding等）。
5. wordVec_150d 为利用本数据集训练的词向量。

### my_code

my_code 中保存所有代码：
1. checSentence.py 检查 predata.txt 中的每一句话是否至少含有一个trigger。
2. combine_data.py 生成 combination.xml。
3. load_polyglot.py 载入polyglot词向量。
4. preprocess.py 利用 predata.txt 生成 train_data.pkl。
5. segSentences.py 分词以训练词向量。
6. models 模型。

### data

data 中保存有 ACE2005 中文数据集以及所用到的一些其他数据。

### data_en

data_en 中保存有 ACE2005 英文数据集
