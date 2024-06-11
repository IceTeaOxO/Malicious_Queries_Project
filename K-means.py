import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 讀取資料
df = pd.read_csv('new_data.csv')

# 使用TF-IDF來轉換url
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['url'])

# 使用K-means來做分群
k = 2  # 假設我們想要分成2群
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# 得到分群結果
labels = kmeans.predict(X)

# 將分群結果加入到原始資料中
df['cluster'] = labels

# 隨機抽樣
sample_df = df.sample(n=1000)  # 假設我們抽樣10000筆資料

# 使用TF-IDF來轉換url
X_sample = vectorizer.transform(sample_df['url'])

# 使用K-means來做分群
kmeans_sample = KMeans(n_clusters=k)
kmeans_sample.fit(X_sample)

# 得到分群結果
labels_sample = kmeans_sample.predict(X_sample)

# 使用PCA來降維
pca = PCA(n_components=2)
X_2d_sample = pca.fit_transform(X_sample.toarray())

colors = ['red' if x == 'bad' else 'blue' for x in sample_df['label']]

# 繪製分群結果
plt.scatter(X_2d_sample[:, 0], X_2d_sample[:, 1], c=colors)
plt.title('K-means Clustering with 2 dimensions')
# 儲存圖片
plt.savefig('kmeans_sample.png')
plt.show()

