# 用于找出数据中事件trigger索引值出现错误的文档
from tqdm import tqdm
from bs4 import BeautifulSoup
from my_code.en.utils import parseDirs
filePath = [
    '../../data/ace_2005_td_v7/data/English/bc/adj',
    '../../data/ace_2005_td_v7/data/English/bn/adj',
    '../../data/ace_2005_td_v7/data/English/cts/adj',
    '../../data/ace_2005_td_v7/data/English/nw/adj',
    '../../data/ace_2005_td_v7/data/English/un/adj',
    '../../data/ace_2005_td_v7/data/English/wl/adj',
]
searchPattern = '.sgm'

files = parseDirs(filePath, searchPattern=searchPattern)

for file in tqdm(files):
    apfFile = file.replace('.sgm', '.apf.xml')

    with open(apfFile, 'r') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    # print(soup.prettify())

    events = soup.find_all('event')
    for e in events:
        emention = e.find('event_mention')

        ldc_scope = emention.find('ldc_scope').find('charseq')
        ldc_text = ldc_scope.get_text()
        ldc_start = int(ldc_scope.get('start'))
        ldc_end = int(ldc_scope.get('end'))

        anchor = emention.find('anchor').find('charseq')
        anchor_start = int(anchor.get('start'))
        anchor_end = int(anchor.get('end'))
        anchor_word = anchor.get_text()

        start = anchor_start - ldc_start
        end = anchor_end - ldc_start

        # print(ldc_text[start: end+1])
        # print(anchor_word)
        # print(ldc_text[start: end+1] == anchor_word)
        if ldc_text[start: end+1] != anchor_word:
            print()
            print('文件中的索引值出现偏差：', file)
            print('索引结果：', ldc_text[start: end+1])
            print('实际结果：', anchor_word)
            break
        # print()
    # exit()