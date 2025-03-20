import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# 自訂測試資料集：假設 test 資料夾內只有影像檔案，檔名即為識別 id
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


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs, writer, patience=20):
    best_acc = 0.0
    best_model_wts = None
    patience_counter = 0  # early stopping counter

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # 每個 epoch 包含訓練 (train) 與驗證 (val) 兩個階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 訓練模式
            else:
                model.eval()   # 驗證模式

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 將每個 phase 的 metrics 寫入 TensorBoard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # 儲存訓練階段的 checkpoint (儲存至 log 目錄)
            if phase == 'train':
                checkpoint_path = os.path.join(
                    writer.log_dir, f'epoch_{epoch}.pth')
                torch.save(model.state_dict(), checkpoint_path)
            else:
                # 驗證階段更新最佳模型並應用早停法
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    patience_counter = 0  # 若有提升，重置 patience
                else:
                    patience_counter += 1
                    print(f'No improvement for {patience_counter} epoch(s).')

        scheduler.step()
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

        # 若驗證表現連續多個 epoch 沒有改善，則提前停止訓練
        if patience_counter >= patience:
            print(
                f'Early stopping triggered after {patience} epochs without improvement.')
            break

    print('訓練完成')
    print(f'最佳驗證準確率: {best_acc:.4f}')
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model


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
                predictions.append({'image_name': image_name.replace(
                    ".jpg", ""), 'pred_label': class_mapping[int(pred)] if class_mapping else int(pred)})

    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    print(f'預測結果已儲存至 {output_file}')


def main():
    parser = argparse.ArgumentParser(description='影像分類訓練程式 - 使用預訓練模型')
    parser.add_argument('--train-dir', type=str,
                        required=True, help='訓練資料夾路徑 (例如：data/train)')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='驗證資料夾路徑 (例如：data/val)')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='測試資料夾路徑 (例如：data/test)')
    parser.add_argument('--exp-name', type=str, default='exp1',
                        help='實驗名稱 (TensorBoard log 目錄名稱)')
    parser.add_argument('--epochs', type=int, default=60, help='訓練的 epoch 數')
    parser.add_argument('--batch-size', type=int, default=256, help='batch 大小')
    parser.add_argument('--lr', type=float, default=0.01, help='初始學習率')
    parser.add_argument('--weight-decay', type=float,
                        default=1e-4, help='優化器 weight decay')
    parser.add_argument('--step-size', type=int,
                        default=20, help='學習率調度器的 step size')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='學習率調度器的 gamma 參數')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='若設置，則凍結預訓練模型的 backbone 參數，只微調最後一層')
    args = parser.parse_args()

    # 使用 arguments 指定的實驗名稱建立 TensorBoard SummaryWriter
    log_dir = os.path.join('runs', args.exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用裝置：{device}')

    # 定義 data augmentation 的文字說明
    da_description = (
        "Train Data Augmentation:\n"
        "- RandomResizedCrop(224, scale=(0.5, 1.0)): 隨機裁剪並調整大小，較大範圍的隨機裁剪\n"
        "- RandomHorizontalFlip(): 隨機左右翻轉\n"
        "- RandomVerticalFlip(): 隨機上下翻轉\n"
        "- RandomRotation(40): 隨機旋轉 ±30 度\n"
        "- ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.3): 調整亮度、對比、飽和度及色調\n"
        "- RandomPerspective(distortion_scale=0.5, p=0.5): 隨機透視變換\n"
        "- RandomErasing(p=0.5): 隨機遮擋部分區域"
    )

    # 記錄超參數設定：包含 data augmentation 說明
    hparams = {
        'exp_name': args.exp_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'step_size': args.step_size,
        'gamma': args.gamma,
        'freeze_backbone': args.freeze_backbone,
        'model': 'ResNet101 with Dropout',
        'data_augmentation': da_description
    }
    writer.add_text('Hyperparameters', json.dumps(hparams, indent=4))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=4)

    # Data augmentation 設定：訓練階段增加更多隨機變換
    # 定義更強的 Data augmentation 設定：訓練階段增加更多隨機變換
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # 擴大隨機裁剪範圍
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # 新增：垂直翻轉
            transforms.RandomRotation(40),     # 將旋轉角度由15擴大到30度
            transforms.ColorJitter(
                brightness=0.7, contrast=0.7, saturation=0.7, hue=0.3),  # 更強的顏色抖動
            transforms.RandomPerspective(
                distortion_scale=0.5, p=0.5),  # 新增：隨機透視變換
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5)  # 新增：隨機遮擋部分區域
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    train_dataset = datasets.ImageFolder(
        args.train_dir, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(
        args.val_dir, transform=data_transforms['val'])
    test_dataset = TestDataset(
        args.test_dir, transform=data_transforms['test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8),
        'val': DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    }
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 使用預訓練的 ResNet101 模型，並在最後全連接層前加入 Dropout
    model = models.resnet101(pretrained=models.ResNet50_Weights.IMAGENET1K_V2)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # dropout rate 可根據需要調整
        nn.Linear(num_features, len(train_dataset.classes))
    )
    print("Data classes:", train_dataset.classes)

    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # 只微調最後的全連接層（含 dropout）
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Backbone 凍結，僅微調最後一層")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma)

    # 訓練模型，並使用早停法防止過擬合
    model = train_model(model, criterion, optimizer, scheduler,
                        dataloaders, device, args.epochs, writer, patience=20)

    # 將最佳模型權重儲存到 log 目錄中
    best_model_path = os.path.join(log_dir, 'best_model.pth')
    torch.save(model.state_dict(), best_model_path)
    print(f'模型權重已儲存至 {best_model_path}')

    output_file = os.path.join(log_dir, 'prediction.csv')
    generate_predictions(model, test_loader, device,
                         output_file=output_file, class_mapping=train_dataset.classes)
    writer.close()


if __name__ == '__main__':
    main()
