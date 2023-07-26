import pandas as pd

df = pd.read_csv("DataForTrain.csv", encoding='utf-8')

# 在这里认为好评是5，中评是3，4，差评是1，2

high_overalls_reviews = df.loc[df['overall'] == 5, 'reviewText'].tolist()
medium_overalls_reviews = df.loc[df['overall'].isin([3, 4]), 'reviewText'].tolist()
low_overalls_reviews = df.loc[df['overall'].isin([1, 2]), 'reviewText'].tolist()

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