# 引入所需的函式庫
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
# 讀取資料
df = pd.read_csv('data.csv')

# 定義DistilBert的tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# 定義權重
weights = [84, 16]
class_weights = torch.FloatTensor(weights).to(device)

# 定義優化器和損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# loss_fn = torch.nn.CrossEntropyLoss().to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

def evaluate(predictions, true_vals):
    # 將預測結果轉換為numpy array並取出最大機率的類別
    preds_flat = np.argmax(predictions, axis=1).flatten()

    # 將真實標籤轉換為numpy array
    labels_flat = true_vals.flatten()

    # 計算混淆矩陣
    cm = confusion_matrix(labels_flat, preds_flat)

    # 計算評估指標
    accuracy = accuracy_score(labels_flat, preds_flat)
    precision = precision_score(labels_flat, preds_flat)
    recall = recall_score(labels_flat, preds_flat)
    f1 = f1_score(labels_flat, preds_flat)

    print('Confusion Matrix:')
    print(cm)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

# 訓練模型
for epoch in range(EPOCHS):
    train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
    # 切換模型為評估模式
    model.eval()
    
    # 初始化記錄預測結果和真實標籤的列表
    predictions, true_vals = [], []
    
    # 進行預測
    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            predictions.append(logits)
            true_vals.append(labels)
    
    predictions = torch.cat(predictions, dim=0)
    true_vals = torch.cat(true_vals, dim=0)
    
    # 評估模型的性能
    evaluate(predictions.cpu().numpy(), true_vals.cpu().numpy())

# 儲存模型
torch.save(model.state_dict(), 'model.pth')


# 預測測試集
# predictions, true_vals = evaluate_model(model, test_data_loader, device)

