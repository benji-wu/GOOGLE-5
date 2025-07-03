# 🚀 安裝依賴（只需第一次）
# pip install pandas numpy nltk sentence-transformers scikit-learn

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === 🔧 下載 nltk 資源（第一次執行需要）
nltk.download('punkt')
nltk.download('stopwords')

# === 📁 1. 讀取並清理 IMDb 電影資料 ===
print("📂 讀取並清理資料...")
df = pd.read_csv("IMDb_Top_1000.csv")
df = df[['Series_Title', 'Genre', 'Overview']].rename(columns={
    'Series_Title': 'Title',
    'Overview': 'Plot'
})

stop_words = set(stopwords.words('english'))
def preprocess(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

df['Clean_Plot'] = df['Plot'].apply(preprocess)
df.to_csv("clean_movie_dataset.csv", index=False)
print("✅ 清理後資料已儲存為 'clean_movie_dataset.csv'")

# === 🤖 2. 將文字轉換為嵌入向量 (Embeddings) ===
print("🔄 轉換劇情文字為向量...")
plots = df['Clean_Plot'][:50].tolist()  # 處理前 50 筆
titles = df['Title'][:50].tolist()
original_plots = df['Plot'][:50].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(plots, show_progress_bar=True)
np.save("movie_embeddings.npy", embeddings)
print("✅ 向量已儲存為 'movie_embeddings.npy'")

# === 🔍 3. 相似度搜尋與推薦函數 ===
def recommend_movies_by_index(index, top_n=5):
    query_vec = embeddings[index].reshape(1, -1)
    similarity_scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = similarity_scores.argsort()[::-1][:top_n + 1]  # 包含自己

    print(f"\n🎬 Top {top_n} similar movies to: {titles[index]}")
    for i in top_indices[1:top_n+1]:  # 排除自己
        print(f"  • {titles[i]}  (Similarity: {similarity_scores[i]:.4f})")

# ✅ 測試推薦功能：第 0 筆電影的 Top 5 相似推薦
recommend_movies_by_index(0)
