import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import joblib

df = pd.read_csv("DataForTrain.csv", encoding='utf-8')
df = df.sample(frac=1.0).reset_index(drop=True)

x=df['reviewText'] #自变量
y=df['overall']  #因变量

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1) #划分测试集和训练集

# 词袋模型
bow_vectorizer = CountVectorizer(max_df=0.80, min_df=2)

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Define classifiers
Knn = KNeighborsClassifier()
Lr = LogisticRegression()
Rf = RandomForestClassifier()
Svm = SGDClassifier()
Nb = MultinomialNB()

# Define vectorizer
bow_vectorizer = CountVectorizer()

# Data preprocessing
# Assuming you have x_train, x_test, y_train, y_test data ready

# Initialize classifiers and their names
classifiers = [Knn, Lr, Rf, Svm, Nb]
classifier_names = ['KNN', 'Logistic Regression', 'Random Forest', 'SVM', 'Naive Bayes']

# Create a list to store accuracy scores
accuracy_scores = []

# Create a directory to save evaluation results if it doesn't exist
output_folder = "output/Evaluation_Results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each classifier, train, evaluate, and save results
for classifier, classifier_name in zip(classifiers, classifier_names):
    # Train the classifier
    pipe = make_pipeline(bow_vectorizer, classifier)
    pipe.fit(x_train, y_train)

    # Perform k-fold cross-validation and get accuracy scores
    accuracy_scores.append(cross_val_score(pipe, x_train, y_train, cv=5, scoring='accuracy').mean())

    # Predict on test data
    y_pred = pipe.predict(x_test)

    # Evaluate and save the classification report to a text file
    evaluation_results = classification_report(y_test, y_pred)
    output_file_path = os.path.join(output_folder, f"{classifier_name}_Evaluation.txt")
    with open(output_file_path, 'w') as output_file:
        output_file.write(evaluation_results)

    # Save the trained model
    joblib.dump(pipe, f"{classifier_name}_Model.pkl")

# Visualize accuracy
plt.figure(figsize=(10, 6))
plt.bar(classifier_names, accuracy_scores)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Classifier Accuracy')
plt.ylim(min(accuracy_scores) - 0.05, max(accuracy_scores) + 0.05)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the accuracy plot
accuracy_plot_path = os.path.join(output_folder, "Classifier_Accuracy.png")
plt.savefig(accuracy_plot_path)

plt.show()


