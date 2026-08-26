import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import langextract as lx
import textwrap
import json 
# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_parquet(
    "hf://datasets/gentaiscool/codemixqa/data/test-00000-of-00001.parquet"
)

# ---------------------------
# Prompt
# ---------------------------
prompt = textwrap.dedent("""\
Segment the question into minimal, non-overlapping spans in order of appearance.

STRICT RULES:
- Use exact substrings from the question.
- Do NOT paraphrase.
- Do NOT merge unrelated concepts into one span.
- Spans must be as small as possible while preserving meaning.
- Every character of the question must belong to exactly one span.
- Do NOT hallucinate or invent text.

Classify each span as:

core_keyword:
    Content-bearing units required to identify the answer.
    Includes:
        - Named entities
        - Titles
        - Specific years or date ranges
        - Key nouns
        - Main lexical verbs
        - Specific numerical quantities

filler:
    Functional or grammatical material.
    Includes:
        - Question words (Which, Kaunsi, Kis, etc.)
        - Case markers and postpositions
        - Auxiliary verbs
        - Determiners
        - Discourse markers
        - Inflectional suffixes
                         
Return output strictly in valid JSON format:

{
  "extractions": [
    {"class": "...", "text": "..."},
    ...
  ]
}

Do not include any explanation or additional text.
Only output valid JSON.
""")

# ---------------------------
# Few-shot examples
# ---------------------------
examples = [
    lx.data.ExampleData(
        text="Kaunsi former First Lady ne 1977 se 1978 tak Met Gala ki co-chair ke roop mein serve kiya?",
        extractions=[
            lx.data.Extraction("filler", "Kaunsi"),
            lx.data.Extraction("core_keyword", "former First Lady"),
            lx.data.Extraction("filler", "ne"),
            lx.data.Extraction("core_keyword", "1977 se 1978"),
            lx.data.Extraction("filler", "tak"),
            lx.data.Extraction("core_keyword", "Met Gala"),
            lx.data.Extraction("core_keyword", "co-chair"),
            lx.data.Extraction("filler", "ki ke roop mein"),
            lx.data.Extraction("core_keyword", "serve"),
            lx.data.Extraction("filler", "kiya?"),
        ],
    ),
    lx.data.ExampleData(
        text="Which baseball field-এ Central Illinois Collegiate League-এর Bluff City Bombers 1998 থেকে 2004 পর্যন্ত তাদের home games খেলত?",
        extractions=[
            lx.data.Extraction("filler", "Which"),
            lx.data.Extraction("core_keyword", "baseball field"),
            lx.data.Extraction("filler", "-এ"),
            lx.data.Extraction("core_keyword", "Central Illinois Collegiate League"),
            lx.data.Extraction("filler", "-এর"),
            lx.data.Extraction("core_keyword", "Bluff City Bombers"),
            lx.data.Extraction("core_keyword", "1998 থেকে 2004"),
            lx.data.Extraction("core_keyword", "home games"),
            lx.data.Extraction("filler", "তাদের খেলত?"),
        ],
    ),
    lx.data.ExampleData(
    text="Kis year mein Rabindranath Tagore ne Nobel Prize jeeta?",
    extractions=[
        lx.data.Extraction("filler", "Kis"),
        lx.data.Extraction("core_keyword", "year"),
        lx.data.Extraction("filler", "mein"),
        lx.data.Extraction("core_keyword", "Rabindranath Tagore"),
        lx.data.Extraction("filler", "ne"),
        lx.data.Extraction("core_keyword", "Nobel Prize"),
        lx.data.Extraction("core_keyword", "jeeta"),
        lx.data.Extraction("filler", "?"),
    ],
)
]

# ---------------------------
# Helper: English detection
# ---------------------------
import re
def is_english_span(text):
    letters = re.findall(r"[A-Za-z]", text)
    return len(letters) > 0 and len(letters) / max(len(text), 1) > 0.6

def english_core_ratio(extractions):
    core = [e for e in extractions if e.extraction_class == "core_keyword"]
    if not core:
        return np.nan
    
    english_core = [e for e in core if is_english_span(e.extraction_text)]
    return len(english_core) / len(core)

# ---------------------------
# Run extraction
# ---------------------------
results = []
ratios = []

with open("langextract_codemixqa1_segmented.jsonl", "w", encoding="utf-8") as f:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["language"]=="Hindi" or row["language"]=="Bengali":
            result = lx.extract(
                text_or_documents=row["problem"],
                prompt_description=prompt,
                examples=examples,
                model_id="phi3",
            )
            record = {
                "id": row["id"],
                "problem": row["problem"],
                "language": row["language"],
                "extractions": [
                    {
                        "class": e.extraction_class,
                        "text": e.extraction_text
                    }
                    for e in result.extractions
                ],
                "english_core_ratio": english_core_ratio(result.extractions)
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
            f.flush()

