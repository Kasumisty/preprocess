# 合并所有数据
import os
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
err_recod = '../../data_en/datacheck.txt'
searchPattern = '.sgm'
savePath = '../../data_en/combination_en.xml'

def get_err_dirs(file):
    with open(file, 'r') as f:
        return [line.strip() for line in f ]

files = set(parseDirs(filePath, searchPattern=searchPattern))
err_files = set(get_err_dirs(err_recod))
files -= err_files

fullData = '<?xml version="1.0"?>\n<document>\n'
for file in tqdm(files):
    apfFile = file.replace('.sgm', '.apf.xml')
    with open(apfFile, 'r') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
    events = soup.find_all('event')

    if not events:
        continue

    # 不能用event.prettify()
    tmp = ''.join([str(event) for event in events])
    base = os.path.basename(file)
    fullData += '<doc id="' + os.path.splitext(os.path.splitext(base)[0])[0] + '">\n'
    fullData += tmp
    fullData += '</doc>\n'
fullData += '</document>'

with open(savePath, 'w') as f:
    f.write(fullData)