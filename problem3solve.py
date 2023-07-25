import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

import pandas as pd
import re

nltk.download('averaged_perceptron_tagger',download_dir='./stopwords_dir')
nltk.download('stopwords',download_dir='./stopwords_dir')
nltk.download('punkt',download_dir='./stopwords_dir')
nltk.download('wordnet',download_dir='./stopwords_dir')

nltk.data.path.append('./stopwords_dir')

wnl = WordNetLemmatizer()
# USdict = enchant.Dict("en_US")
stoplist = set(stopwords.words('english'))

def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return (vTEXT)


def get_word(text):
    text = text.replace('.', ' ')
    rawwords = nltk.word_tokenize(text)
    words = [word.lower() for word in rawwords if not str.isdigit(word) and len(word) > 2]
    return words


def get_pos_word(words):
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    words = pos_tag(words)

    pos_word = [wnl.lemmatize(tag[0], pos=get_wordnet_pos(tag[1]) or wordnet.NOUN) for tag in words]

    # 停用词过滤
    cleanwords = [word for word in pos_word if word not in stoplist]

    return cleanwords

import json
from collections import defaultdict

if __name__ == '__main__':
    filedata = pd.read_excel('Appendix I.xlsx')
    # 读到所有的json数据中需要的部分
    reviews = []
    overalls = []
    for index, row in filedata.iterrows():
        json_data = row[0]  # 提取第一列的JSON字符串
        data = json.loads(json_data)  # 解析JSON数据
        reviews.append(data["reviewText"])  # 提取reviewText字段的内容
        overalls.append(data["overall"])
        
    dataset = {'reviewText':reviews,'overall':overalls}
    df = pd.DataFrame(dataset)
    
	#去除链接
    df['reviewText'] = df['reviewText'].apply(remove_urls)
	
    # 分词
    df['reviewText'] = df['reviewText'].apply(get_word)
	
    # 文本处理结果
    df['reviewText'] = df['reviewText'].apply(get_pos_word)
	
    # 删除tweets中的空列表
    df = df[~(df['reviewText'].str.len() == 0)]
	
    # 转换字符串
    df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join(x))
	
    # 打乱顺序
    df = df.sample(frac=1.0).reset_index(drop=True)
	
	#保存文本
    df.to_csv('DataForTrain.csv',encoding='utf-8')
