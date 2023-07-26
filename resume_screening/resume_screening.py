import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

resumeDataSet = pd.read_csv("UpdatedResumeDataSet.csv")
resumeDataSet["cleaned_resume"] = ""
print(resumeDataSet.head())
print(resumeDataSet.isnull().sum())
print(resumeDataSet["Category"].unique())
print(resumeDataSet["Category"].value_counts())

# now let's visualize the number of categories in the dataset

'''
import seaborn as sns

#plt.figure(figsize=(15,15)) #boyut belirlemek için çok da gerekli değil olmasa da olur
#plt.xticks(rotation=45) # aşağıdaki sayıların kaç derece ile duracağını gösteriyor boş bir metod
sns.countplot(y="Category", data=resumeDataSet)
plt.show()
'''
'''
#pasta grafik gösterimi 
from matplotlib.gridspec import GridSpec
targetCounts = resumeDataSet['Category'].value_counts()
targetLabels  = resumeDataSet['Category'].unique()
# Make square figures and axes
plt.figure(1, figsize=(25,25))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, 3)]
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()

'''

import re

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
    
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

'''
Yukardakiyle aynı işlemi yapan kod 
import re

def clean_resume(resume_text):
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text) 
    resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
    return resume_text
    
resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(clean_resume)

'''
'''
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
totalWords =[]
Sentences = resumeDataSet['Resume'].values
cleanedSentences = ""
for i in range(0,160):
    cleanedText = cleanResume(Sentences[i])
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)
    
wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
'''

from sklearn.preprocessing import LabelEncoder

var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])



#sklearn.feature_extraction.text.TfidfVectorizer:
#Bu modül, metin verilerini TF-IDF (Term Frequency-Inverse Document Frequency) vektörleri olarak temsil etmek için kullanılır. 
# Metin verileri, makine öğrenimi algoritmaları tarafından doğrudan işlenemez, bu nedenle metin verilerini sayısal vektörlerle temsil etmek gerekir. 
# TF-IDF, bir belgedeki terimlerin önemini belirlemek için sık kullanılan bir metrik olup, terim frekansını (TF) ve ters belge frekansını (IDF) birleştirir. 
# TfidfVectorizer, metin verilerini TF-IDF vektörleri olarak temsil eder ve makine öğrenimi modellerine uygun hale getirir.

#scipy.sparse.hstack:
#Bu modül, seyrek (sparse) matrisleri yatay yönde birleştirmek için kullanılır. 
# Metin verilerini vektörleştirirken çoğunlukla seyrek matrisler elde edilir. 
# Seyrek matrislerde, çoğu öğe sıfırdır ve sadece sınırlı sayıda öğe önemli değerlere sahiptir. 
# Makine öğrenimi algoritmaları için matrislerin düzgün ve tutarlı bir boyuta sahip olması önemlidir. 
# hstack işlevi, seyrek matrisleri yan yana birleştirerek, metin verileriyle başka özelliklerin vektörleştirilmiş hallerini birleştirmek için kullanılabilir. 
# Böylece, metin verilerini diğer özelliklerle birleştirerek tam bir veri matrisi oluşturulabilir ve makine öğrenimi modellerine beslemek için uygun hale getirilebilir.


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english', max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print ("Feature completed .....")

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

# now let's train the model and print the classification report

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction))) 