import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image

# 自訂測試資料集：假設 test 資料夾內只有影像檔，檔名即為識別 id


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(test_dir))  # 請確保資料夾內只有影像檔

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.test_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_name


def evaluate_model(model, dataloader, device):
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    accuracy = running_corrects.double() / total
    return accuracy.item()


def generate_predictions(model, test_loader, device,
                         output_file='prediction.csv'):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, image_names in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            # 使用 os.path.splitext 移除檔案副檔名
            for image_name, pred in zip(image_names, preds):
                name_without_ext = os.path.splitext(image_name)[0]
                predictions.append(
                    {'image_name': name_without_ext,
                     'pred_label': CLASSES[int(pred)]})
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    print(f'預測結果已儲存至 {output_file}')


def plot_confusion_matrix(
        model, val_loader, device, class_names,
        output_file="confusion_matrix.jpg"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # 混淆矩陣計算仍使用最大值預測
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    # 儲存圖檔成 jpg 格式
    plt.savefig(output_file, format='jpg')
    plt.close()
    print(f'Confusion matrix saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='模型評估程式：計算 train/val accuracy 並對 test 資料集進行預測')
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型權重檔案路徑 (例如：runs/exp1/best_model.pth)')
    parser.add_argument('--train-dir', type=str,
                        required=True, help='訓練資料夾路徑 (例如：data/train)')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='驗證資料夾路徑 (例如：data/val)')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='測試資料夾路徑 (例如：data/test)')
    parser.add_argument('--batch-size', type=int, default=256, help='batch 大小')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader 的 workers 數量')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用裝置：{device}')

    # 定義統一的 evaluation transform (不進行隨機 augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 載入 train 與 val 資料集 (用非隨機的 eval_transform)
    train_dataset = datasets.ImageFolder(
        args.train_dir, transform=eval_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=eval_transform)
    test_dataset = TestDataset(args.test_dir, transform=eval_transform)

    # train_loader = DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # 初始化與訓練時相同的模型架構 (ResNet18 with Dropout)
    model = models.resnet101(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, len(train_dataset.classes))
    )
    model = model.to(device)

    # 載入訓練好的權重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f'成功載入模型權重：{args.model_path}')
    print("資料集類別：", train_dataset.classes)
    global CLASSES
    CLASSES = train_dataset.classes

    # 計算 train 與 val 的 accuracy
    # train_acc = evaluate_model(model, train_loader, device)
    # val_acc = evaluate_model(model, val_loader, device)
    # print(f"Train Accuracy: {train_acc:.4f}")
    # print(f"Val Accuracy: {val_acc:.4f}")

    # 對 test 資料集產生預測結果，並儲存為 prediction.csv
    generate_predictions(model, test_loader, device,
                         output_file='prediction.csv')

    # 使用 validation set 畫出 confusion matrix 並儲存成 jpg 檔
    class_names = val_dataset.classes
    plot_confusion_matrix(model, val_loader, device, class_names)


if __name__ == '__main__':
    main()
