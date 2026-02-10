from datasets import load_dataset
from transformers import BertTokenizer
import torch
import re

def clean_wikitext(text: str) -> str:
    """
    Remove Wikipedia markup and formatting artifacts from WikiText-2.
    
    Removes:
    - Section headers (= Title =, == Section ==, === Subsection ===, etc.)
    - @ symbols (article markers)
    - # symbols (list markers)
    - Isolated punctuation and special characters
    - Multiple consecutive spaces/newlines
    - Leading/trailing whitespace
    """
    if not text or not text.strip():
        return ""
    
    # Remove section headers: = Title =, == Section ==, === Subsection ===, etc.
    # Matches 1+ equals on both sides with optional spaces
    text = re.sub(r'\s*=+\s*.+?\s*=+\s*', ' ', text)
    
    # Remove @ symbols (often used as article markers)
    text = re.sub(r'@+', '', text)
    
    # Remove # symbols when used as list markers or standalone
    # Keep # when part of actual words (e.g., "C#" but unlikely in WikiText)
    text = re.sub(r'\s#+\s+', ' ', text)  # " # " -> " "
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # Beginning of line
    text = re.sub(r'\s+#+$', '', text, flags=re.MULTILINE)  # End of line
    
    # Remove common Wikipedia markup artifacts
    text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Remove {{template}} markup
    text = re.sub(r'\[\[([^|\]]+\|)?([^\]]+)\]\]', r'\2', text)  # [[link|text]] -> text
    
    # Remove isolated special characters (periods, hyphens, etc. on their own line)
    text = re.sub(r'^\s*[.,;:\-_]+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove excessive punctuation clusters (3+ repeated punctuation)
    text = re.sub(r'([.,;:!?\-_])\1{2,}', r'\1', text)
    
    # Normalize whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
    
    # Clean up spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

class WikiTextDataset:
    def __init__(self, split="train", max_length=32, clean_markup=True, dataset_version="wikitext-103-raw-v1"):
        """
        WikiText dataset loader with optional markup cleaning.
        
        Args:
            split: "train", "validation", or "test"
            max_length: Maximum sequence length
            clean_markup: Whether to remove Wikipedia markup (=, @, #, etc.)
            dataset_version: "wikitext-2-raw-v1" (2M tokens) or "wikitext-103-raw-v1" (100M tokens)
        """
        self.dataset = load_dataset("Salesforce/wikitext", dataset_version)[split]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.clean_markup = clean_markup
        self.dataset_version = dataset_version

        # Filter empty texts and optionally clean markup
        if self.clean_markup:
            self.texts = [clean_wikitext(t) for t in self.dataset["text"] if len(t.strip()) > 0]
            # Remove any texts that became empty after cleaning
            self.texts = [t for t in self.texts if len(t.strip()) > 0]
            dataset_name = "WikiText-2" if "2" in dataset_version else "WikiText-103"
            print(f"📝 {dataset_name} {split}: {len(self.texts)} samples after cleaning")
        else:
            self.texts = [t for t in self.dataset["text"] if len(t.strip()) > 0]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return enc["input_ids"].squeeze(0)