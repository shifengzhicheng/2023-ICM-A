import pandas as pd
import json

# 读取xlsx文件
df = pd.read_excel("Appendix II.xlsx")  # 替换成附件中数据的路径

review_texts = []

for index, row in df.iterrows():
    json_data = row[0]  # 提取第一列的JSON字符串
    data = json.loads(json_data)  # 解析JSON数据
    review_texts.append(data["reviewText"])  # 提取reviewText字段的内容

file_path = "reviews.txt"

# 将所有评论写入txt文件
with open(file_path, 'w', encoding='utf-8') as file:
    for review in review_texts:
        file.write(review + '\n')
        



