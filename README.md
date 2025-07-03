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
🧠 Understanding Recommender Systems
Recommender systems are algorithms designed to suggest relevant items to users. There are two main types of recommendation strategies:

🔹 Content-Based Filtering
This method recommends items based on the similarity between the content of the items and the user's preferences. It uses the features of the items (e.g., movie plots, song lyrics, etc.) to find and recommend similar items. For text-based data, techniques like TF-IDF vectorization or sentence embeddings are used to compute similarity.

Advantages:

Works well even with limited user data.

Can recommend rare or niche items if the content is relevant.

Disadvantages:

May lead to over-specialization (i.e., always recommending similar content).

Doesn’t consider what other users liked.

🔸 Collaborative Filtering
This method recommends items based on user interactions and the preferences of other users. It assumes that if two users have similar tastes, they will enjoy similar items in the future. It requires user-item interaction data (e.g., ratings, likes, views).

Advantages:

Can discover surprising or serendipitous content.

Doesn't require metadata or features of the items.

Disadvantages:

Struggles with the "cold start" problem (new users or items).

Needs large-scale interaction data to be effective.

✅ Chosen Recommender Approach
For this project, I have chosen Content-Based Filtering because the dataset is entirely text-based (movie plot summaries), and we do not have access to user interaction data. This approach allows us to use Natural Language Processing techniques such as tokenization, TF-IDF, and cosine similarity to find and recommend movies with similar themes or storylines.

The dataset used is IMDb_Top_1000.csv, which includes over 1000 movie entries with fields like title, genre, and plot. The plot field has been preprocessed and saved in a cleaned format (clean_movie_dataset.csv) for further modeling

