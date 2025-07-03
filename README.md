# 🚀 第一次使用請先執行以下安裝：
# pip install pandas numpy nltk sentence-transformers

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sentence_transformers import SentenceTransformer

# 📌 NLTK 初始化（僅需第一次）
nltk.download('punkt')
nltk.download('stopwords')

# === 📁 1. 讀取與清理資料 ===
# 請下載並確認檔案路徑為 IMDb_Top_1000.csv
df = pd.read_csv("IMDb_Top_1000.csv")

# 選取欄位並重新命名
df = df[['Series_Title', 'Genre', 'Overview']].rename(columns={
    'Series_Title': 'Title',
    'Overview': 'Plot'
})

# 清理文字內容（去除停用詞與標點）
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

df['Clean_Plot'] = df['Plot'].astype(str).apply(preprocess)

# 儲存清理後的 CSV（可選）
df.to_csv("clean_movie_dataset.csv", index=False)
print("✅ Cleaned movie dataset saved as 'clean_movie_dataset.csv'")

# === 🤖 2. 使用 SentenceTransformer 產生向量嵌入 ===
# 取前 50 筆劇情摘要（符合任務最低需求）
plots = df['Clean_Plot'][:50].tolist()

# 載入預訓練模型（384 維度輸出）
model = SentenceTransformer('all-MiniLM-L6-v2')

# 執行轉換
embeddings = model.encode(plots, show_progress_bar=True)

# 儲存為 .npy 格式以便後續推薦使用
np.save("movie_embeddings.npy", embeddings)
print("✅ Embeddings generated and saved as 'movie_embeddings.npy'")

