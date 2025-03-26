import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from collections import Counter
from resnest.torch import resnest101
# from resnest.torch import resnest50

# 若有多卡可考慮加速
torch.backends.cudnn.benchmark = True


class TestDataset(Dataset):
    # 自訂測試資料集：假設 test 資料夾內只有影像檔案，檔名即為識別 id

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


def mixup_data(x, y, alpha=0.4, device='cuda'):
    # 實作 mixup (可選)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_model(model, criterion, optimizer, scheduler, dataloaders,
                device, num_epochs, writer, use_mixup=False,
                mixup_alpha=0.4, patience=10):
    # 訓緓與驗證模型函式，內建 mixup 與 early stopping

    best_acc = 0.0
    best_model_wts = None
    patience_counter = 0  # early stopping 計數器

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        # 每個 epoch 包含訓練與驗證階段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                if phase == 'train' and use_mixup:
                    # 使用 mixup 資料擴充
                    inputs, targets_a, targets_b, lam = mixup_data(
                        inputs, labels, alpha=mixup_alpha, device=device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if phase == 'train' and use_mixup:
                        loss = lam * \
                            criterion(outputs, targets_a) + \
                            (1 - lam) * criterion(outputs, targets_b)
                        # 預測時取兩組預測中較大者作為正確率評估（近似）
                        _, preds = torch.max(outputs, 1)
                    else:
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # 計算正確率時以原始標籤為準
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)

            # 訓練階段存檔 (可選)
            if phase == 'train':
                checkpoint_path = os.path.join(
                    writer.log_dir, f'epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
            else:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f'No improvement for {patience_counter} epoch(s).')

        scheduler.step()
        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

        if patience_counter >= patience:
            print(
                f'Early stopping triggered after {patience} epochs \
                    without improvement.')
            break

    print('Training complete')
    print(f'Best Val Accuracy: {best_acc:.4f}')
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model


def generate_predictions(
        model, test_loader, device, output_file='prediction.csv',
        class_mapping=None):
    # 產生測試預測
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, image_names in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            for image_name, pred in zip(image_names, preds):
                predictions.append(
                    {'image_name': image_name.replace(".jpg", ""),
                     'pred_label': class_mapping[int(pred)]
                     if class_mapping else int(pred)})
    import pandas as pd
    df = pd.DataFrame(predictions)
    df.to_csv(output_file, index=False)
    print(f'Predictions saved to {output_file}')


def main():
    parser = argparse.ArgumentParser(
        description='ResNeSt Image Classification Training')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Path to training data folder (e.g., data/train)')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Path to validation data folder (e.g., data/val)')
    parser.add_argument('--test-dir', type=str, required=True,
                        help='Path to test data folder (e.g., data/test)')
    parser.add_argument('--exp-name', type=str, default='exp_resnest',
                        help='Experiment name (TensorBoard log folder name)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                        default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float,
                        default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate in the classifier head')
    parser.add_argument('--use-mixup', action='store_true',
                        help='Use mixup augmentation during training')
    parser.add_argument('--mixup-alpha', type=float,
                        default=0.4, help='Alpha parameter for mixup')
    parser.add_argument(
        '--freeze-backbone', action='store_true',
        help='Freeze backbone parameters (only train classifier)')
    args = parser.parse_args()

    log_dir = os.path.join('runs', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data augmentation 設定
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5)
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
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
        'train': DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=8),
        'val': DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=8)
    }
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # 建立 ResNeSt101 模型並修改最後全連接層：包含 dropout
    model = resnest101(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(args.dropout),
        nn.Linear(num_features, len(train_dataset.classes))
    )
    print("Data classes:", train_dataset.classes)

    # 若選擇凍結 backbone，僅更新最後全連接層
    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        print("Backbone frozen; only fine-tuning classifier head.")

    model = model.to(device)

    # 結合 train 與 val 的標籤數據
    combined_targets = train_dataset.targets + val_dataset.targets

    # 計算各類別在 train+val 中的數量
    combined_counts = Counter(combined_targets)
    print("Combined class counts:", combined_counts)

    # 計算總樣本數（train + val）
    total_samples = len(combined_targets)
    num_classes = len(train_dataset.classes)

    # 計算每個類別的權重：使用總樣本數除以該類別數量
    class_weights = [total_samples / combined_counts[i]
                     for i in range(num_classes)]
    class_weights = torch.tensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    # 使用 AdamW 優化器 (SGD could be used as well)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # 使用 Cosine Annealing 調度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # 記錄超參數與設定
    hparams = {
        'exp_name': args.exp_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'use_mixup': args.use_mixup,
        'mixup_alpha': args.mixup_alpha,
        'freeze_backbone': args.freeze_backbone,
        'model': 'ResNeSt101'
    }
    writer.add_text('Hyperparameters', json.dumps(hparams, indent=4))
    with open(os.path.join(log_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=4)

    # 訓練模型
    model = train_model(model, criterion, optimizer, scheduler,
                        dataloaders, device, args.epochs, writer,
                        use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
                        patience=args.patience)

    best_model_path = os.path.join(log_dir, 'best_model.pth')
    torch.save(model.state_dict(), best_model_path)
    print(f'Model weights saved to {best_model_path}')

    # 產生測試集預測結果
    output_file = os.path.join(log_dir, 'prediction.csv')
    generate_predictions(
        model, test_loader, device, output_file=output_file,
        class_mapping=train_dataset.classes)
    writer.close()


if __name__ == '__main__':
    main()
