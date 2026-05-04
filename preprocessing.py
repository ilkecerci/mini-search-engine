import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # 1. Case folding
    text = text.lower()
    
    # 2. Punctuation and symbol removal
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenization
    tokens = text.split()
    
    # 4. Stop word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Stemming
    stemmer = PorterStemmer()
    final_terms = [stemmer.stem(word) for word in filtered_tokens]
    
    return final_terms
