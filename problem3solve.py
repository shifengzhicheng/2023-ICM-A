# 载入必要库
import jieba
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt 
import pyecharts.options as opts
from pyecharts.charts import WordCloud
from pyecharts.charts import Bar
import re
#logging
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel('Appendix I.xlsx')

# 移除含有缺失值的行
data.dropna(axis=0,inplace=True)
#查看去除缺失值后的行和列
data.shape

def remove_url(src):
    # 去除标点符号、数字、字母
    vTEXT = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】╮ ￣ ▽ ￣ ╭\\～⊙％；①（）：《》？“”‘’！[\\]^_`{|}~\s]+', "", src)
    return vTEXT


cutted = []
for row in data.values:
        text = remove_url(str(row[0])) #去除文本中的标点符号、数字、字母
        raw_words = (' '.join(jieba.cut(text)))#分词,并用空格进行分隔
        cutted.append(raw_words)

cutted_array = np.array(cutted)

# 生成新数据文件，Comment字段为分词后的内容
data_cutted = pd.DataFrame({
    'Comment': cutted_array,
    'Class': data['Class']
})

data_cutted.head()#查看分词后的数据集

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.data.path.append('./stopwords_dir')
# extract stopwords
stop_words = set(stopwords.words('english'))
# 初始化WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

for i in stop_words[:100]:#读前100个停用词
    print(i,end='')
    
#设定停用词文件,在统计关键词的时候，过滤停用词
import jieba.analyse
jieba.analyse.set_stop_words('./dataset/stopwords.txt')