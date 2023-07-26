import spacy
import pytextrank

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Load the pre-processed reviews from the file
with open('Pre_Processed_reviews.txt', 'r', encoding='utf-8') as f:
    all_texts = f.readlines()

# Add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")

# Function to process a chunk of texts using TextRank
def process_chunk(chunk_texts):
    processed_phrases = []
    docs = list(nlp.pipe(chunk_texts))
    for doc in docs:
        for phrase in doc._.phrases:
            processed_phrases.append((phrase.text, phrase.rank, phrase.count, phrase.chunks))
    return processed_phrases

# Process the text in smaller chunks using nlp.pipe()
chunk_size = 10000  # Set an appropriate chunk size for your system's memory capacity

# Split the all_texts into chunks
text_chunks = [all_texts[i:i + chunk_size] for i in range(0, len(all_texts), chunk_size)]

# Process each chunk and collect the results
all_processed_phrases = []
for chunk in text_chunks:
    processed_phrases = process_chunk(chunk)
    all_processed_phrases.extend(processed_phrases)

import pandas as pd

# Create a list to store the data
data = []

# Store the extracted phrases, ranks, counts, and chunks in the data list
for phrase_text, rank, count, chunks in all_processed_phrases:
    data.append([phrase_text, rank, count, chunks])

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['Phrase', 'Rank', 'Count', 'Chunks'])

# Write the DataFrame to an Excel file
excel_file_path = 'NLP_output.xlsx'
df.to_excel(excel_file_path, index=False)

print("Data written to Excel file:", excel_file_path)


