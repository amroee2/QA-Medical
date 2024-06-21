# pip install rank_bm25
# pip install nltk
# pip install arabic_reshaper
# pip install python-bidi
# pip install qalsadi
# pip install textblob



import string
from textblob import TextBlob
import re
import qalsadi.lemmatizer as ql
lemmatizer = ql.Lemmatizer()
import pyarabic.araby as araby
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Define the stopwords list for Arabic
arabic_stopwords = set(stopwords.words('arabic'))

# stopwords = [arabic_reshaper.reshape(s) for s in arabic_stopwords]
# stopwords=set(stopwords)

def remove_arabic_stopwords(text, stopwords=arabic_stopwords):
        # Split the text into words
        words = text.split()
        
        # Remove punctuation from words and filter out stopwords
        table = str.maketrans('', '', string.punctuation)
        filtered_words = [word.translate(table) for word in words if word.translate(table) not in stopwords]
        
        # Join the filtered words back into a string
        filtered_text = ' '.join(filtered_words)
        
        return filtered_text

class ArabicTextPreprocessor:
    def __init__(self):
        pass

    def remove_punctuation(self,text):
        punc=string.punctuation+"،"
        text= text.translate(str.maketrans('', '', punc))
        return text
    def adding_a_space_between_a_word_and_a_punctuation(self,text):   #pourquoi cette fonction?
        text = re.sub(r"([?.!,¿،])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        return text
    
    def lemmatize(self,text):
        lemmas= [lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(lemmas)
    def normalizeArabic(self,text):
        text = text.strip()
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
        text = re.sub(noise, '', text)
        text = re.sub(r'(.)\1+', r"\1\1", text) # Remove longation
        return araby.strip_tashkeel(text)

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    def add_start_and_end_tokens(self,text):
        return '<end> '+text.strip()+' <start>'
    

textpreprocessor=ArabicTextPreprocessor()

#Preprocessing train, test and val data:
def Preprocess_question(qst1):
  print(f'Question1:\n{qst1}')
  qst1=textpreprocessor.remove_punctuation(qst1)
  print(f'Question1 without punctuation:\n{qst1}')
  qst1=remove_arabic_stopwords(qst1)
  print(f'Question1 without punctuation and without stopwords:\n{qst1}')
  qst1=textpreprocessor.lemmatize(qst1)
  print(f'Question1 lemmatized:\n{qst1}')
  qst1=textpreprocessor.normalizeArabic(qst1)
  print(f'Question1 normalised:\n{qst1}')
  return qst1

Preprocess_question("""ما هي مسببات الاكتئاب

  """)

# Load the CSV file
final = pd.read_csv('FinalTrain.csv')




# Function to normalize Arabic text
def normalize_arabic(text):
    # Normalize different forms of the same letter
    text = text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    # Remove diacritics
    text = ''.join([c for c in text if c not in "ًٌٍَُِّْ"])
    return text

# Tokenize the documents

tokenized_corpus = []
for doc in final['question']:
    doc = normalize_arabic(doc)
    tokens = [word for word in word_tokenize(doc.lower()) if word not in arabic_stopwords]
    tokenized_corpus.append(tokens)

# Initialize the BM25 model
bm25 = BM25Okapi(tokenized_corpus)

# Function to retrieve the most relevant documents
def retrieve_context(query, bm25, df, top_n=1):
    query = normalize_arabic(query)
    tokenized_query = [word for word in word_tokenize(query.lower()) if word not in arabic_stopwords]
    scores = bm25.get_scores(tokenized_query)
    best_doc_indices = scores.argsort()[-top_n:][::-1]
    return df.iloc[best_doc_indices]['answer'].tolist()

# Retrieve the most relevant documents
# relevant_docs = retrieve_context(query, bm25, final, top_n=1)

def AnswerQuestion(question):
    question = Preprocess_question(question)
    print(question)
    answer = retrieve_context(question, bm25, final, top_n=1)
    return answer





# Example query
query = "ما هي اعراض مرض السكري ؟"




# print(AnswerQuestion(query))
# print(f"no of docs = {len(relevant_docs)}")
# for doc in relevant_docs:
#     reshaped_text = arabic_reshaper.reshape(doc)
# print(f"Relevant document: {reshaped_text}")

print(Preprocess_question(query))

print(AnswerQuestion(query))