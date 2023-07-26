import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import joblib

df = pd.read_csv("DataForTrain.csv", encoding='utf-8')
df = df.sample(frac=1.0).reset_index(drop=True)

x=df['reviewText'] #自变量
y=df['overall']  #因变量

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) #划分测试集和训练集

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.80, min_df=2)

# SVM Classifier
Svm = SGDClassifier()
pipe = make_pipeline(tfidf_vectorizer, Svm)
pipe.fit(x_train, y_train)

y_pred = pipe.predict(x_test) #预测

from sklearn import metrics

print(metrics.classification_report(y_test, y_pred)) #评估

joblib.dump(pipe, 'Bayes.pkl') #保存模型
