# 载入必要库
import json
import pandas as pd

#读入数据集
df = pd.read_excel('Appendix I.xlsx')

review_texts = []

overalls = []

for index, row in df.iterrows():
    json_data = row[0]  # 提取第一列的JSON字符串
    data = json.loads(json_data)  # 解析JSON数据
    review_texts.append(data["reviewText"])
    overalls.append(data["overall"])

import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords',download_dir='./stopwords_dir')
# nltk.download('punkt',download_dir='./stopwords_dir')
# nltk.download('wordnet',download_dir='./stopwords_dir')

nltk.data.path.append('./stopwords_dir')
# extract stopwords
stop_words = set(stopwords.words('english'))
# 初始化WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # lower
    text = text.lower()
    # decompose
    text = re.sub(r'[^\w\s]', '', text)
    # split
    words = nltk.word_tokenize(text)
    
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # reviews stem
    stemmer = SnowballStemmer('english')
    processed_words = [stemmer.stem(word) for word in filtered_words]
    # recombined
    processed_text = ' '.join(processed_words)
    return processed_text

processed_review_texts = []
for review in review_texts:
    processed_review_texts.append(preprocess_text(review))

# 在这里认为好评是5，中评是3，4，差评是1，2

high_overalls_reviews = []
medium_overalls_reviews = []
low_overalls_reviews = []


for review, overall in zip(processed_review_texts, overalls):
    if overall in (1, 2):
        low_overalls_reviews.append(review)
    elif overall in (3,4):
        medium_overalls_reviews.append(review)
    elif overall == 5:
        high_overalls_reviews.append(review)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

# 创建 WordCloud 对象
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis')

# 分别处理三个类别的评论并生成词云图
categories = ['Low Overalls', 'Medium Overalls', 'High Overalls']
review_lists = [low_overalls_reviews, medium_overalls_reviews, high_overalls_reviews]

output_folder = "output/Problem3"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i, reviews in enumerate(review_lists):
    # 将评论列表拼接成一个字符串
    text = ' '.join(reviews)

    # 生成词云图
    cloud = wordcloud.generate(text)

    # 绘制词云图
    plt.figure(figsize=(10, 5))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(categories[i] + ' Word Cloud')
    
    wordcloud_output_path = os.path.join(output_folder, f"WordCloud_{categories[i]}.png")
    plt.savefig(wordcloud_output_path)
    
    # 输出最高频的30个词并绘制柱状图
    word_freq = Counter(text.split())
    top_30_words = word_freq.most_common(30)
    top_words, frequencies = zip(*top_30_words)  # Unzip the word-frequency pairs

    plt.figure(figsize=(12, 6))
    plt.bar(top_words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f"Top 30 words in {categories[i]}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存词云图和柱状图到文件
    
    bar_chart_output_path = os.path.join(output_folder, f"WordFreq_{categories[i]}.png")
    plt.savefig(bar_chart_output_path)

plt.show()