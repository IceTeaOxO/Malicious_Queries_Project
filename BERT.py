# 引入所需的函式庫
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

# 讀取資料
df = pd.read_csv('data.csv')

# 定義BERT的tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定義數據集
class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_len):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, item):
        url = str(self.urls[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            url,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'url_text': url,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 切分訓練集和測試集
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# 創建訓練集和測試集的DataLoader
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# 定義模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 定義優化器和損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# 訓練模型
for epoch in range(EPOCHS):
    train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))

# 預測測試集
predictions, true_vals = evaluate_model(model, test_data_loader, device)