import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# 第一次執行請先下載資源
nltk.download('punkt')
nltk.download('stopwords')

# 讀取資料
df = pd.read_csv("IMDb_Top_1000.csv")  # 請確認檔案路徑

# 選擇需要的欄位
df = df[['Series_Title', 'Genre', 'Overview']].rename(columns={
    'Series_Title': 'Title',
    'Overview': 'Plot'
})

# 資料前處理：清理劇情摘要
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

df['Clean_Plot'] = df['Plot'].astype(str).apply(preprocess)

# 儲存處理後資料
df.to_csv("clean_movie_dataset.csv", index=False)
print("✅ Cleaned dataset saved as clean_movie_dataset.csv")


