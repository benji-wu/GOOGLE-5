# 🚀 安裝依賴（僅需一次）
# pip install pandas numpy nltk sentence-transformers scikit-learn

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === 🔧 初始化 nltk
nltk.download('punkt')
nltk.download('stopwords')

# === 📂 讀取電影資料與嵌入向量 ===
df = pd.read_csv("clean_movie_dataset.csv")
titles = df['Title'][:50].tolist()
plots = df['Plot'][:50].tolist()
embeddings = np.load("movie_embeddings.npy")

# === 🤖 載入 Sentence-BERT 模型
model = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# === 🔍 定義推薦函數 ===
def recommend_from_user_input(user_input, top_n=5):
    cleaned = preprocess(user_input)
    query_vec = model.encode([cleaned])[0].reshape(1, -1)
    similarity_scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    print("\n📽️ Top 5 Recommended Movies:")
    for i in top_indices:
        print(f"• {titles[i]}  (Similarity: {similarity_scores[i]:.4f})")

# === 🖥️ CLI 介面 ===
if __name__ == "__main__":
    print("🎬 Welcome to the Movie Recommender System!")
    print("Type in a short movie description or plot, and get 5 similar films.\n")

    while True:
        user_input = input("📝 Enter a plot description (or type 'exit' to quit):\n> ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break
        recommend_from_user_input(user_input)
        print("\n" + "-"*50 + "\n")
