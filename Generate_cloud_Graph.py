file_path = "Pre_Processed_reviews.txt"

review_texts = []

# 将所有评论写入txt文件
with open(file_path, 'r', encoding='utf-8') as file:
    for review in file:
        review_texts.append(review)
        
# review_texts是包含所有reviewText的列表
all_reviews_text = ' '.join(review_texts)
# 输出文件
import os
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.data.path.append('./stopwords_dir')

words = word_tokenize(all_reviews_text)  
filtered_words = [word for word in words if word.isalpha()]
 
# 使用Counter统计词频
word_freq_counter = Counter(filtered_words)
sorted_word_freq = sorted(word_freq_counter.items(), key=lambda x: x[1], reverse=True)

# 绘制前50个词汇的条形图
top_50_words = sorted_word_freq[:50]
words, frequencies = zip(*top_50_words)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequencies')
plt.title('Top 50 Words by Frequency')
plt.xticks(rotation=90)  # 将x轴标签旋转90度，使得词汇能够显示完整
plt.tight_layout()  # 调整布局，防止标签被截断
plt.savefig('output/word_freq/word_freq_bar_chart_2.png', bbox_inches='tight')
     
from wordcloud import WordCloud

# 创建WordCloud对象
wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(all_reviews_text)

# 输出到文件夹output/WordCloud中
output_folder = "output/WordCloud"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_file_path = os.path.join(output_folder, "wordcloud_2.png")

# 将单词云图保存为图像文件
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")  # 关闭坐标轴显示
plt.savefig(output_file_path, bbox_inches='tight', dpi=300)