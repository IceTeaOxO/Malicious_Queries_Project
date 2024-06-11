import pandas as pd

# 因為資料總共有約45萬筆，先將資料做整合
# 讀取資料
data = pd.read_csv('data/data.csv')
data2 = pd.read_csv('data/data2.csv')
# 將data2加入到data
data = pd.concat([data, data2])
print(data.head(5))
# 刪除重複資料
data = data.drop_duplicates()
# 存檔
data.to_csv('data.csv', index=False)
# 約剩下40萬筆資料