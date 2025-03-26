import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from resnest.torch import resnest101


class TestDataset(Dataset):
    # 自訂測試資料集，假設 test 資料夾內只有影像檔案
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


def generate_predictions(
        model, test_loader, device, output_file='prediction.csv',
        class_mapping=None, with_prob=False):
    # 產生測試預測並存成 CSV
    # 當 with_prob 為 True 時：
    #   - 若最大機率 >= 0.7 則採用該預測；否則使用 multinomial 隨機抽樣
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, image_names in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if with_prob:
                # 計算 softmax 機率分布
                # shape: (batch_size, num_classes)
                probs = torch.softmax(outputs, dim=1)
                # 取得每個樣本的最大機率及其對應 label
                max_probs, max_labels = torch.max(probs, dim=1)
                final_preds = []
                for i in range(probs.size(0)):
                    if max_probs[i] < 0.7:
                        # 若最大機率小於 0.7，則依據該樣本的機率分布使用 multinomial 隨機抽樣
                        sampled_label = torch.multinomial(
                            probs[i].unsqueeze(0), num_samples=1).item()
                        final_preds.append(sampled_label)
                    else:
                        # 否則採用最大值
                        final_preds.append(max_labels[i].item())
                preds = torch.tensor(final_preds)
            else:
                # 直接取最大機率的預測
                _, preds = torch.max(outputs, dim=1)
            for image_name, pred in zip(image_names, preds):
                predictions.append(
                    {'image_name': image_name.replace(".jpg", ""),
                     'pred_label': class_mapping[int(pred)]
                     if class_mapping else int(pred)})
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    print(f'Predictions saved to {output_file}')


def plot_confusion_matrix(model, val_loader, device, class_names,
                          output_file="confusion_matrix.jpg"):
    # 計算並繪製 validation set 的 confusion matrix，並儲存成 jpg 檔
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
    plt.savefig(output_file, format='jpg')
    plt.close()
    print(f'Confusion matrix saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for ResNeSt model")
    parser.add_argument('--weights', type=str, required=True,
                        help="Path to the .pth weights file")
    parser.add_argument('--val-dir', default='data/val', type=str)
    parser.add_argument('--test-dir', default='data/test', type=str,
                        help="Path to the test images directory")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size for inference")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument(
        '--with-prob', action='store_true',
        help="If set, for predictions with max probability < 0.7 sample based on distribution; otherwise use max")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 建立 ResNeSt101 模型架構，注意：pretrained 設定為 False 以便載入自己的權重
    model = resnest101(pretrained=False)
    num_features = model.fc.in_features
    # 定義類別，並依據類別數量調整最後全連接層
    class_list = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17',
                  '18', '19', '2', '20', '21', '22', '23', '24', '25', '26',
                  '27', '28', '29', '3', '30', '31', '32', '33', '34', '35',
                  '36', '37', '38', '39', '4', '40', '41', '42', '43', '44',
                  '45', '46', '47', '48', '49', '5', '50', '51', '52', '53',
                  '54', '55', '56', '57', '58', '59', '6', '60', '61', '62',
                  '63', '64', '65', '66', '67', '68', '69', '7', '70', '71',
                  '72', '73', '74', '75', '76', '77', '78', '79', '8', '80',
                  '81', '82', '83', '84', '85', '86', '87', '88', '89', '9',
                  '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
    num_classes = len(class_list)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )

    # 載入權重檔，若權重檔由多卡訓練請加 map_location
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 定義測試資料轉換 (與訓練時使用的轉換相同)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 建立 validation 資料集與 dataloader (使用 torchvision.datasets.ImageFolder)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # 建立測試資料集與 dataloader
    test_dataset = TestDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 產生 prediction.csv 檔案
    output_file = 'prediction.csv'
    generate_predictions(model, test_loader, device,
                         output_file=output_file, class_mapping=class_list,
                         with_prob=args.with_prob)

    # 使用 validation set 畫出 confusion matrix 並儲存成 jpg 檔
    class_names = val_dataset.classes
    plot_confusion_matrix(model, val_loader, device, class_names)


if __name__ == '__main__':
    main()
