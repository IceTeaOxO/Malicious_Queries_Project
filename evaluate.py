# 引入所需的函式庫
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch

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

# 使用模型進行預測
predictions, true_vals = evaluate_model(model, test_data_loader, device)

# 評估模型的性能
evaluate(predictions, true_vals)