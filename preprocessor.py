"""
Text preprocessing module for Bilingual (English + Hindi + Hinglish) SMS Spam Detection.
Preserves Unicode/Devanagari characters while cleaning noise, URLs, emails, and currency symbols.
"""

import re
import html

def clean_sms_text(text: str) -> str:
    """
    Cleans raw SMS text for feature extraction:
    1. Unescapes HTML entities (e.g., &amp; -> &)
    2. Lowercases English characters (Devanagari has no case)
    3. Normalizes web URLs to a token 'url'
    4. Normalizes email addresses to a token 'email'
    5. Normalizes currency symbols (₹, $, £, €) to 'money'
    6. Removes punctuation while preserving alphanumeric characters and Devanagari script (U+0900 to U+097F)
    7. Collapses consecutive whitespace into a single space
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Unescape HTML entities
    text = html.unescape(text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    
    # 4. Replace email addresses
    text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', ' email ', text)
    
    # 5. Replace currency symbols
    text = re.sub(r'[₹$£€]', ' money ', text)
    
    # 6. Remove punctuation and special symbols, keeping:
    #    - \w (alphanumeric English, digits, underscore)
    #    - \s (whitespace)
    #    - \u0900-\u097F (Devanagari Unicode block: consonants, independent vowels, matras, viramas, anusvara, etc.)
    text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    
    # 7. Normalize multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
