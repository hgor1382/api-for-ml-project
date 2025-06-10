import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from hazm import Normalizer, word_tokenize
import re

# Load model
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=7):
        super(MLPClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier()
model.load_state_dict(torch.load("best_mlp_model.pt", map_location=device))
model.to(device)
model.eval()

# Tokenizer and Roberta
try:
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", force_download=True)
    roberta = AutoModel.from_pretrained("xlm-roberta-base", force_download=True)
except Exception as e:
    st.error(f"Error in loading RoBERTa: {e}")

roberta.to(device)
roberta.eval()

# Normalization
normalizer = Normalizer()

def clean_text(text):
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    for p, e in zip(persian_digits, english_digits):
        text = text.replace(p, e)
    text = text.replace('\u200c', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = roberta(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding

# UI
st.title("دسته‌بندی احساسات متون فارسی")
user_input = st.text_area("متن خود را وارد کنید:")

if st.button("پیش‌بینی احساس"):
    cleaned = clean_text(user_input)
    emb = get_embedding(cleaned)
    with torch.no_grad():
        output = model(emb)
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

    emotions = ['ANGRY', 'FEAR', 'HAPPY', 'HATE', 'OTHER', 'SAD', 'SURPRISE']
    top = np.argsort(probs)[::-1]

    st.write(f"### برچسب پیش‌بینی‌شده: **{emotions[np.argmax(probs)]}**")
    st.bar_chart({emotions[i]: probs[i] for i in top})
