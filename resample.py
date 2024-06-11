import pandas as pd

# 讀取原始數據
df = pd.read_csv('data.csv')

# 隨機選取20000筆數據
df_sample = df.sample(n=5000)

# 將選取的數據儲存為新的CSV檔案
df_sample.to_csv('new_data.csv', index=False)