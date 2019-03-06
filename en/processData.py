# 将数据处理为易于后续处理的格式保存
import nltk
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict

combine_data = '../../data_en/combination_en.xml'
savePath = '../../data_en/pred_en.txt'
with open(combine_data, 'r') as f:
    soup = BeautifulSoup(f.read(), 'lxml')

save_file = open(savePath, 'w', encoding='utf-8')
docs = soup.find_all('doc')
for doc in tqdm(docs):
    events = doc.find_all('event')

    save_file.write('docid=' + doc.get('id') + '\n')

    sentences = {}
    trigger_idx = defaultdict(dict)

    for e in events:
        eventType = e.get('type')
        eventSubType = e.get('subtype')
        emention = e.find('event_mention')

        ldc_scope = emention.find('ldc_scope').find('charseq')
        ldc_text = ldc_scope.get_text()
        ldc_start = int(ldc_scope.get('start'))

        anchor = emention.find('anchor').find('charseq')
        anchor_start = int(anchor.get('start'))
        anchor_end = int(anchor.get('end'))
        anchor_text = anchor.get_text()

        start = int(anchor_start - ldc_start)
        end = int(anchor_end - ldc_start + 1)

        if ldc_start not in sentences:
            sentences[ldc_start] = ldc_text

        trigger_idx[ldc_start][start] = (eventType, eventSubType)
        # print(ldc_text[start: end])
        # print(anchor_text)
        # if ldc_text[start: end] != anchor_text:
        #     print(doc.get('id'))
        # print(sentences)
        # print(trigger_idx)

    for ldc_start_i, sent in sentences.items():
        idx = 0
        tokens = nltk.word_tokenize(sent)
        for w in tokens:
            # print()
            # print(sent)

            # 修正偏移量
            i = sent.find(w)
            idx += i
            if idx in trigger_idx[ldc_start_i]:
                save_file.write(w + '\t' +
                                trigger_idx[ldc_start_i][idx][0] + '\t' +
                                trigger_idx[ldc_start_i][idx][1] + '\n')
            else:
                save_file.write(w + '\n')

            # print('idx', idx)
            idx += len(w)

            sent = sent[len(w) + i:]

            # print(w)
            # print(sent)
            # print()
        save_file.write('\n')


