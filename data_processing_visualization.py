# 引入視覺化套件
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 讀取資料
data = pd.read_csv('new_data.csv')
# 顯示data的label分布
print(data['label'].value_counts())
# 繪製圓餅圖
data['label'].value_counts().plot.pie(autopct='%1.1f%%')
# 儲存圖片
plt.savefig('label_pie_chart.png')
plt.show()

