import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/SMSSpamCollection.tsv", sep="\t", names = ['label','text'])
df.head()

df.shape

for i in range(20):
  print(df.iloc[i,-1])
  print()

# Data Cleaning :

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
nltk.download('punkt')

sn = SnowballStemmer("english")
stop = set(stopwords.words('english'))

def clean_text(sms):
  sms = sms.lower()
  sms = re.sub("[^a-z0-9]", ' ', sms)
  sms = nltk.word_tokenize(sms)
  sms = [t for t in sms if len(t)>1]
  sms = [sn.stem(word) for word in sms if word not in stop]
  sms = " ".join(sms)
  return sms

clean_text("Get is ...  are .... playing they UNLIMITED <!!!!>,,,,,????? Free data play   100 GB ....///// at Rs. 0")


df['clean_text'] = df['text'].apply(clean_text)
df.head()

# WordCloud :

from wordcloud import WordCloud

df['label'] == 'ham'

hamdata = df[df['label'] == 'ham']
hamdata 

hamdata = df[df['label'] == 'ham']
hamdata = hamdata['clean_text'].values
hamdata

' '.join(hamdata) # make all data from ham text into 1 single text;

def wordcloud(data):
  words = ' '.join(data)
  wc = WordCloud(width=1000, height=500, background_color='white')
  wc = wc.generate(words)
  plt.axis("off")
  plt.imshow(wc)

print("Ham Data words : ")
wordcloud(hamdata)

spamdata = df[df['label'] == 'spam']
spamdata = spamdata['clean_text']

print("Spam Data words : ")
wordcloud(spamdata)

# Featuization : 

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(df['clean_text']).toarray()

X.shape
X

y = pd.get_dummies(df['label'])
y = y['spam'].values
y

# Model Building : 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_test, y_pred

pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

# Model Evaluation :

print("Accuracy on Training Data :")
print(model.score(x_train, y_train))


print("Accuracy on Testing Data :")
print(model.score(x_test, y_test))

from sklearn.metrics import confusion_matrix, f1_score, classification_report

confusion_matrix(y_test, y_pred)

y_test.shape

print(classification_report(y_test, y_pred))

f1_score(y_test, y_pred) # 0.95





















# Detailed Exlaination given below : 


import numpy as np  # For numerical computations and array operations
import pandas as pd  # For data manipulation and analysis, especially for handling tabular data
import matplotlib.pyplot as plt  # For data visualization

# Load the dataset
df = pd.read_csv("/content/SMSSpamCollection.tsv", sep="\t", names=['label', 'text'])  # Read the TSV file into a DataFrame with specified column names
df.head()  # Display the first few rows of the dataset to understand its structure

df.shape  # Get the shape of the DataFrame (number of rows and columns)

# Print the first 20 messages
for i in range(20):
    print(df.iloc[i, -1])  # Print the text of each message
    print()  # Print a new line for better readability

# Data Cleaning
import re  # Regular expressions for text processing
import nltk  # Natural Language Toolkit for text processing tasks
from nltk.corpus import stopwords  # For accessing a list of common stopwords
from nltk.stem.snowball import SnowballStemmer  # For stemming words to their root form

# Download necessary NLTK data
nltk.download('stopwords')  # Download the list of stopwords
nltk.download('punkt')       # Download the tokenizer

# Initialize stemmer and stopwords
sn = SnowballStemmer("english")  # Create an English language stemmer
stop = set(stopwords.words('english'))  # Create a set of English stopwords for filtering

# Function to clean the text messages
def clean_text(sms):
    sms = sms.lower()  # Convert text to lowercase for uniformity
    sms = re.sub("[^a-z0-9]", ' ', sms)  # Remove all non-alphanumeric characters
    sms = nltk.word_tokenize(sms)  # Tokenize the cleaned text into words
    sms = [t for t in sms if len(t) > 1]  # Remove tokens with only one character
    sms = [sn.stem(word) for word in sms if word not in stop]  # Stem the words and remove stopwords
    sms = " ".join(sms)  # Join the processed tokens back into a single string
    return sms  # Return the cleaned text

# Example of cleaning a sample message
clean_text("Get is ... are .... playing they UNLIMITED <!!!!>,,,,,????? Free data play 100 GB ....///// at Rs. 0")

# Apply the cleaning function to the 'text' column
df['clean_text'] = df['text'].apply(clean_text)  # Create a new column with cleaned text
df.head()  # Display the cleaned DataFrame

# Generate WordCloud for 'ham' and 'spam'
hamdata = df[df['label'] == 'ham']  # Filter the DataFrame to get only 'ham' messages
hamdata = hamdata['clean_text'].values  # Extract the cleaned text of 'ham' messages

# Function to create and display a WordCloud
def wordcloud(data):
    words = ' '.join(data)  # Join all the cleaned 'ham' texts into a single string
    wc = WordCloud(width=1000, height=500, background_color='white').generate(words)  # Generate the WordCloud
    plt.axis("off")  # Turn off the axes for a cleaner look
    plt.imshow(wc)  # Display the generated WordCloud

print("Ham Data words:")  # Print header
wordcloud(hamdata)  # Display WordCloud for 'ham' messages

spamdata = df[df['label'] == 'spam']  # Filter the DataFrame to get only 'spam' messages
spamdata = spamdata['clean_text']  # Extract the cleaned text of 'spam' messages

print("Spam Data words:")  # Print header
wordcloud(spamdata)  # Display WordCloud for 'spam' messages

# Featurization using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer  # Importing CountVectorizer for text feature extraction

cv = CountVectorizer(max_features=5000)  # Initialize CountVectorizer to limit to 5000 features
X = cv.fit_transform(df['clean_text']).toarray()  # Fit and transform the cleaned text into feature vectors
X.shape  # Display the shape of the feature matrix

y = pd.get_dummies(df['label'])  # Convert labels into dummy variables for classification
y = y['spam'].values  # Extract the 'spam' column as the target variable (1 for spam, 0 for ham)

# Model Building
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting the dataset
from sklearn.naive_bayes import MultinomialNB  # Import Multinomial Naive Bayes classifier

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # Split data into training and testing sets (80% train, 20% test)

model = MultinomialNB()  # Initialize the Naive Bayes model
model.fit(x_train, y_train)  # Fit the model on the training data

y_pred = model.predict(x_test)  # Predict labels for the test set
pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  # Create a DataFrame to compare actual and predicted labels

# Model Evaluation
print("Accuracy on Training Data:")  # Print accuracy for training data
print(model.score(x_train, y_train))  # Calculate and print accuracy on training data

print("Accuracy on Testing Data:")  # Print accuracy for testing data
print(model.score(x_test, y_test))  # Calculate and print accuracy on testing data

# Confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))  # Display the confusion matrix to visualize performance

print(classification_report(y_test, y_pred))  # Print detailed classification report including precision, recall, and F1 score

print(f"F1 Score: {f1_score(y_test, y_pred)}")  # Print the F1 score, a measure of model accuracy

