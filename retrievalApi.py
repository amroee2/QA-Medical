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
nltk.download('punkt')


arabic_stopwords = set([
    "،","آض","آمينَ","آه","آهاً","آي","أ","أب","أجل","أجمع","أخ","أخذ","أصبح",
    "أضحى","أقبل","أقل","أكثر","ألا","أم","أما","أمامك","أمامكَ","أمسى","أمّا",
    "أن","أنا","أنت","أنتم","أنتما","أنتن","أنتِ","أنشأ","أنّى","أو","أوشك","أولئك",
    "أولئكم","أولاء","أولالك","أوّهْ","أي","أيا","أين","أينما","أيّ","أَنَّ","أََيُّ","أُفٍّ","إذ",
    "إذا","إذاً","إذما","إذن","إلى","إليكم","إليكما","إليكنّ","إليكَ","إلَيْكَ","إلّا","إمّا",
    "إن","إنّما","إي","إياك","إياكم","إياكما","إياكن","إيانا","إياه","إياها","إياهم",
    "إياهما","إياهن","إياي","إيهٍ","إِنَّ","ا","ابتدأ","اثر","اجل","احد","اخرى","اخلولق",
    "اذا","اربعة","ارتدّ","استحال","اطار","اعادة","اعلنت","اف","اكثر","اكد","الألاء",
    "الألى","الا","الاخيرة","الان","الاول","الاولى","التى","التي","الثاني","الثانية",
    "الذاتي","الذى","الذي","الذين","السابق","الف","اللائي","اللاتي","اللتان","اللتيا"
    ,"اللتين","اللذان","اللذين","اللواتي","الماضي","المقبل","الوقت","الى","اليوم",
    "اما","امام","امس","ان","انبرى","انقلب","انه","انها","او","اول","اي","ايار",
    "ايام","ايضا","ب","بات","باسم","بان","بخٍ","برس","بسبب","بسّ","بشكل","بضع","بطآن"
    ,"بعد","بعض","بك","بكم","بكما","بكن","بل","بلى","بما","بماذا","بمن","بن","بنا",
    "به","بها","بي","بيد","بين","بَسْ","بَلْهَ","بِئْسَ","تانِ","تانِك","تبدّل","تجاه","تحوّل",
    "تلقاء","تلك","تلكم","تلكما","تم","تينك","تَيْنِ","تِه","تِي","ثلاثة","ثم","ثمّ","ثمّة",
    "ثُمَّ","جعل","جلل","جميع","جير","حار","حاشا","حاليا","حاي","حتى","حرى","حسب","حم",
    "حوالى","حول","حيث","حيثما","حين","حيَّ","حَبَّذَا","حَتَّى","حَذارِ","خلا","خلال","دون",
    "دونك","ذا","ذات","ذاك","ذانك","ذانِ","ذلك","ذلكم","ذلكما","ذلكن","ذو","ذوا",
    "ذواتا","ذواتي","ذيت","ذينك","ذَيْنِ","ذِه","ذِي","راح","رجع","رويدك","ريث","رُبَّ",
    "زيارة","سبحان","سرعان","سنة","سنوات","سوف","سوى","سَاءَ","سَاءَمَا","شبه","شخصا",
    "شرع","شَتَّانَ","صار","صباح","صفر","صهٍ","صهْ","ضد","ضمن","طاق","طالما","طفق","طَق",
    "ظلّ","عاد","عام","عاما","عامة","عدا","عدة","عدد","عدم","عسى","عشر","عشرة","علق",
    "على","عليك","عليه","عليها","علًّ","عن","عند","عندما","عوض","عين","عَدَسْ","عَمَّا",
    "غدا","غير","ـ","ف","فان","فلان","فو","فى","في","فيم","فيما","فيه","فيها","قال",
    "قام","قبل","قد","قطّ","قلما","قوة","كأنّما","كأين","كأيّ","كأيّن","كاد","كان",
    "كانت","كذا","كذلك","كرب","كل","كلا","كلاهما","كلتا","كلم","كليكما","كليهما",
    "كلّما","كلَّا","كم","كما","كي","كيت","كيف","كيفما","كَأَنَّ","كِخ","لئن","لا","لات",
    "لاسيما","لدن","لدى","لعمر","لقاء","لك","لكم","لكما","لكن","لكنَّما","لكي",
    "لكيلا","للامم","لم","لما","لمّا","لن","لنا","له","لها","لو","لوكالة","لولا",
    "لوما","لي","لَسْتَ","لَسْتُ","لَسْتُم","لَسْتُمَا","لَسْتُنَّ","لَسْتِ","لَسْنَ","لَعَلَّ","لَكِنَّ","لَيْتَ",
    "لَيْسَ","لَيْسَا","لَيْسَتَا","لَيْسَتْ","لَيْسُوا","لَِسْنَا","ما","ماانفك","مابرح","مادام",
    "ماذا","مازال","مافتئ","مايو","متى","مثل","مذ","مساء","مع","معاذ","مقابل",
    "مكانكم","مكانكما","مكانكنّ","مكانَك","مليار","مليون","مما","ممن","من","منذ",
    "منها","مه","مهما","مَنْ","مِن","نحن","نحو","نعم","نفس","نفسه","نهاية","نَخْ",
    "نِعِمّا","نِعْمَ","ها","هاؤم","هاكَ","هاهنا","هبّ","هذا","هذه","هكذا","هل","هلمَّ",
    "هلّا","هم","هما","هن","هنا","هناك","هنالك","هو","هي","هيا","هيت","هيّا","هَؤلاء",
    "هَاتانِ","هَاتَيْنِ","هَاتِه","هَاتِي","هَجْ","هَذا","هَذانِ","هَذَيْنِ","هَذِه","هَذِي","هَيْهَاتَ","و",
    "و6","وا","واحد","واضاف","واضافت","واكد","ان","واهاً","واوضح","وراءَك","في",
    "وقال","وقالت","وقد","وقف","وكان","وكانت","ولا","ولم","ومن","مَن","هو","هي",
    "ويكأنّ","وَيْ","وُشْكَانََ","يكون","يمكن","يوم","ّأيّان","في", "من", "على", "إلى", "و", "ب", "أن", "ما", "لا", "نعم", "نحن", "هو", "هي",
    "كان", "كانت", "ذلك", "هذه", "هذا", "هؤلاء", "الذي", "التي", "أو", "أي", "أين",
    "كل", "بين", "مع", "عند", "إذ", "إذا", "لكن", "لأن", "لقد", "أكثر", "أقل", "نحو",
    "مثل", "عن", "أمام", "بعد", "قبل", "غير", "دون", "كيف", "حتى", "إلا", "حين",
    "أثناء", "فقط", "بينما", "ضمن", "ذلك"
])

stopwords = [arabic_reshaper.reshape(s) for s in arabic_stopwords]
stopwords=set(stopwords)

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
    def remove_stopwords(self,text,stopwords):
        textb=TextBlob(text)  
        words=textb.words
        return " ".join([w for w in words if w not in stopwords])
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
  qst1=textpreprocessor.remove_stopwords(qst1,stopwords)
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

# Example query
query = "ما هي اعراض مرض السكري ؟"



# Retrieve the most relevant documents
# relevant_docs = retrieve_context(query, bm25, final, top_n=1)

def AnswerQuestion(question):
    answer = retrieve_context(query, bm25, final, top_n=1)
    return answer

# print(f"no of docs = {len(relevant_docs)}")
# for doc in relevant_docs:
#     reshaped_text = arabic_reshaper.reshape(doc)
# print(f"Relevant document: {reshaped_text}")

