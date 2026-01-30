from datasets import load_dataset
from transformers import BertTokenizer
import torch

class WikiTextDataset:
    def __init__(self, split="train", max_length=32):
        self.dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[split]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

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