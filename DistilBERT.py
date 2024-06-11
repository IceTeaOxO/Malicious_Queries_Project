import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 讀取資料
df = pd.read_csv('new_data.csv')

# 將標籤轉換為數字
label_map = {'good': 0, 'bad': 1}
df['label'] = df['label'].map(label_map)

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
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'url_text': url,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 定義DataLoader的創建函數
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = URLDataset(
        urls=df.url.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

# 切分訓練集和測試集
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# 設定超參數
MAX_LEN = 128
BATCH_SIZE = 6
EPOCHS = 3
LEARNING_RATE = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 創建訓練集和測試集的DataLoader
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

# 定義模型
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model = model.to(device)

# 定義權重 (根據標籤分佈調整)
weights = [df['label'].value_counts()[1], df['label'].value_counts()[0]]
class_weights = torch.FloatTensor(weights).to(device)

# 定義優化器和損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

# 定義訓練函數
def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)

# 定義評估函數
def evaluate(predictions, true_vals):
    pred_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_vals.flatten()
    
    accuracy = accuracy_score(labels_flat, pred_flat)
    precision = precision_score(labels_flat, pred_flat)
    recall = recall_score(labels_flat, pred_flat)
    f1 = f1_score(labels_flat, pred_flat)
    
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

# 訓練模型
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_acc, train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, None, len(df_train))
    print(f'Train loss {train_loss} accuracy {train_acc}')

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

print(f"模型在GPU上運行: {next(model.parameters()).is_cuda}")
