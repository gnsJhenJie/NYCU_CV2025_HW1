import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from resnest.torch import resnest101  # 請先安裝 resnest 套件

# 自訂測試資料集，假設 test 資料夾內只有影像檔案


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

# 產生測試預測並存成 csv


def generate_predictions(model, test_loader, device, output_file='prediction.csv', class_mapping=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, image_names in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            for image_name, pred in zip(image_names, preds):
                predictions.append({
                    'image_name': image_name.replace(".jpg", ""),
                    'pred_label': class_mapping[int(pred)] if class_mapping else int(pred)
                })
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    print(f'Predictions saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for ResNeSt model")
    parser.add_argument('--weights', type=str, required=True,
                        help="Path to the .pth weights file")
    parser.add_argument('--test-dir', type=str, required=True,
                        help="Path to the test images directory")
    parser.add_argument('--batch-size', type=int, default=256,
                        help="Batch size for inference")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f"Using device: {device}")

    # 建立 ResNeSt101 模型架構，注意：pretrained 設定為 False 以便載入自己的權重
    model = resnest101(pretrained=False)
    num_features = model.fc.in_features
    # 如果有提供類別名稱，則 class 數量依據列表長度，否則使用預設模型輸出維度
    num_classes = len(['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52',
                      '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99'])
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )

    # 載入權重檔，若權重檔由多卡訓練請加 map_location
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 定義測試資料的轉換 (與訓練時使用的 test 轉換相同)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(args.test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 預測結果存成 prediction.csv 到目前目錄
    output_file = 'prediction.csv'
    generate_predictions(model, test_loader, device,
                         output_file=output_file, class_mapping=['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99'])


if __name__ == '__main__':
    main()
