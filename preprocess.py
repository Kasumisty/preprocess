import pickle
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from mycode.load_polyglot import getEmbeddingModel

c = 0

# trigger 不在词向量中: 解决方法——    1.使用其他词向量   2.将trigger加入训练词向量的过程（优先选择此方法）
def lookup(mess, key, dic):
    if key not in dic:
        dic[key] = len(dic)
        print(mess + ':', key, '==>', dic[key])


def parseInst(inst, model, eventTypeDict, maxlen=100, k=64):
    '''

    :param inst: list of instance
    :param model: word2vec model
    :param eventTypeDict:
    :param maxlen: max length of a instance
    :param k:dimension of word vector   词向量维度
    :return:the length of inst, embeddings of inst, trigger infomation
    '''
    triggerInfo = []
    embeddings = np.zeros(shape=[maxlen, k])
    instLen = 0
    for line in inst:
        line = line.split('\t')
        if len(line) > 1:
            lookup('eventType', line[1], eventTypeDict)
            if line[0] not in model:
                print('err', line[0])
                # add unknown trigger words
                embeddings[instLen] = np.random.uniform(low=-0.25, high=0.25, size=k)
                global c
                c += 1
            else:
                embeddings[instLen] = np.array(model[line[0]])
                # print(line[0])
            triggerInfo.append((instLen, eventTypeDict[line[1]]))
            instLen += 1
        else:
            if line[0] in model:
                embeddings[instLen] = np.array(model[line[0]])
            else:
                embeddings[instLen] = np.random.uniform(low=-0.25, high=0.25, size=k)
                # print(line[0])
            instLen += 1
    return instLen, embeddings, triggerInfo

# def getEmbedding(model, inst, candidate=None, maxlen=32, k=150):
#     embeddings = np.zeros(shape=[maxlen, k])
#     count = 0
#     for word in inst:
#         if word in model:
#             try:
#                 embeddings[count] = np.array(model[word])
#                 count += 1
#             except OverflowError:
#                 pass
#     return count, embeddings

if __name__ == '__main__':
    k = 64
    vec_file = '../processed_data/wordVec_150d'
    file = '../processed_data/predata.txt'
    save_path = '../processed_data/train_data.pkl'

    # 载入词向量
    model = getEmbeddingModel()
    # model = Word2Vec.load(vec_file)

    eventTypeDict = defaultdict(int)
    eventTypeDict['None'] = 0
    maxlen = 0
    revs = []
    inst = []
    # c, e = getEmbedding(model, ['中国', ',', '中央'])
    # print(c, e)
    # exit()
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('i'):
                continue
            if line:
                inst += [line]
                continue
            # print(inst)
            # break
            instLen, embeddings, triggerInfo = parseInst(inst, model, eventTypeDict, maxlen=131, k=k)
            # print(triggerInfo)
            # print(eventTypeDict)
            # print(instLen)
            # print(embeddings[:instLen])
            # break

            # 记录最大的样本长度
            if instLen > maxlen:
                maxlen = instLen

            revs.append({'embeddings': embeddings, 'triggerInfo': triggerInfo, 'instLen': instLen})
            inst = []
    print('maxlen of one inst:', maxlen)
    print('num of unknown trigger words:', c)
    with open(save_path, 'wb') as f:
        pickle.dump(revs, f)
