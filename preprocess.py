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
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # decompose
    text = re.sub(r'\W', ' ', text)
    # lower
    text = text.lower()
    # split
    words = nltk.word_tokenize(text)
    
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # reviews stem
    stemmer = SnowballStemmer('english')
    processed_words = [stemmer.stem(word) for word in lemmatized_words]
    # recombined
    processed_text = ' '.join(processed_words)
    return processed_text

file_path = "reviews.txt"

review_texts = []
# read file
with open(file_path, 'r') as file:
    for review in file:
        review_texts.append(preprocess_text(review))

# write file        
new_file_path = "Pre_Processed_reviews.txt"

with open(new_file_path, 'w') as file:
    for review in review_texts:
        file.write(review + '\n')
        