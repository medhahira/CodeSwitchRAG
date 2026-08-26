language_codes = {
    "Toba Batak": "id",            # No dedicated ISO-639-1; Google uses Indonesian
    "Urdu": "ur",
    "Hindi": "hi",
    "Japanese": "ja",
    "French": "fr",
    "Marathi": "mr",
    "Italian": "it",
    "Bengali": "bn",
    "Korean": "ko",
    "Spanish": "es",
    "Indonesian": "id",
    "Simplified Chinese": "zh-cn"
}

from tqdm import tqdm
from googletrans import Translator
import time
import re
import os
import pandas as pd
import os
df = pd.read_parquet("hf://datasets/gentaiscool/codemixqa/data/test-00000-of-00001.parquet")
output_dir_lang = "language_texts"
translator = Translator()
output_dir_translated = "language_texts_translated"
os.makedirs(output_dir_translated, exist_ok=True)

MAX_CHARS = 4500

def clean_text(text):
    """Remove common web noise from scraped text"""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common navigation/footer text patterns
    text = re.sub(r'(Home|Archive|Travel|Index|About Us|©.*)', '', text, flags=re.IGNORECASE)
    return text.strip()

for key, lang_code in language_codes.items():
    print(f"Translating files to {key} ({lang_code})")
    output_dir_translated_lang = os.path.join(output_dir_translated, lang_code)
    os.makedirs(output_dir_translated_lang, exist_ok=True)
    
    for filename in tqdm(os.listdir(output_dir_lang)):
        file_path = os.path.join(output_dir_lang, filename)
        translated_file_path = os.path.join(output_dir_translated_lang, filename)
        
        if os.path.exists(translated_file_path):
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Clean the text first
        text = clean_text(text)
        
        if not text or len(text) < 50:  # Skip very short texts
            continue
            
        try:
            if len(text) > MAX_CHARS:
                chunks = [text[i:i+MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]
                translated_chunks = []
                for chunk in chunks:
                    result = translator.translate(chunk, dest=lang_code)
                    if result and result.text:
                        translated_chunks.append(result.text)
                    time.sleep(0.5)
                translated_text = " ".join(translated_chunks)
            else:
                result = translator.translate(text, dest=lang_code)
                translated_text = result.text if result else ""
            
            if translated_text:
                with open(translated_file_path, "w", encoding="utf-8") as f:
                    f.write(translated_text)
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"\nFailed to translate {filename} to {key}: {e}")
            time.sleep(2)