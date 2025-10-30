"""
Text Cleaning and Preprocessing Module
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextCleaner:
    """
    A class for cleaning and preprocessing text data
    """
    
    def __init__(self, 
                 lowercase=True,
                 remove_urls=True,
                 remove_html=True,
                 remove_emails=True,
                 remove_numbers=True,
                 remove_punctuation=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 stem=False):
        """
        Initialize TextCleaner with preprocessing options
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_html: Remove HTML tags
            remove_emails: Remove email addresses
            remove_numbers: Remove numbers
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove English stopwords
            lemmatize: Apply lemmatization
            stem: Apply stemming (if True, lemmatization is disabled)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize and not stem
        self.stem = stem
        
        # Initialize tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if self.lemmatize else None
        self.stemmer = PorterStemmer() if self.stem else None
    
    def clean_text(self, text):
        """
        Apply all cleaning steps to the text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        if self.remove_html:
            text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatization
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Stemming
        if self.stem:
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        # Join tokens back to string
        cleaned_text = ' '.join(tokens)
        
        # Remove extra whitespace
        cleaned_text = ' '.join(cleaned_text.split())
        
        return cleaned_text
    
    def clean_dataframe(self, df, text_column='text', output_column='cleaned_text'):
        """
        Clean text column in a pandas DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text to clean
            output_column: Name of column for cleaned text
            
        Returns:
            DataFrame with cleaned text column
        """
        df = df.copy()
        df[output_column] = df[text_column].apply(self.clean_text)
        return df


def remove_short_texts(df, text_column='cleaned_text', min_words=10):
    """
    Remove texts with fewer than min_words words
    
    Args:
        df: Input DataFrame
        text_column: Column containing text
        min_words: Minimum number of words required
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    df['word_count'] = df[text_column].str.split().str.len()
    df = df[df['word_count'] >= min_words].reset_index(drop=True)
    return df


def combine_text_columns(df, title_col='title', text_col='text', output_col='full_text'):
    """
    Combine title and text columns
    
    Args:
        df: Input DataFrame
        title_col: Title column name
        text_col: Text column name
        output_col: Output column name
        
    Returns:
        DataFrame with combined text column
    """
    df = df.copy()
    df[output_col] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
    df[output_col] = df[output_col].str.strip()
    return df


if __name__ == "__main__":
    # Example usage
    sample_text = """
    Breaking News! Visit https://example.com for more info.
    Email us at test@email.com. This is a TEST with 123 numbers!!!
    """
    
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(sample_text)
    
    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(cleaned)
