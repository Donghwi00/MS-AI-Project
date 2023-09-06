import argparse
import os
import glob
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torchvision
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt

from torch.optim import lr_scheduler
from albumentations.pytorch import ToTensorV2
from torchvision.models.densenet import densenet121
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from custom_dataset import CustomDataset

class Fashion_Recommendation:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

    def train(self, train_loader, val_loader, epochs, optimizer, criterion, scheduler, start_epoch=0):
        best_val_acc = 0.0
        print("Training.....")

        for epoch in range(start_epoch, epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_acc = 0.0
            val_acc = 0.0

            self.model.train()
            scheduler.step()
            train_loader_iter = tqdm(train_loader, desc=(f"Epoch : {epoch + 1}/{epochs}"), leave=False)

            for index, (data, target) in enumerate(train_loader_iter):
                data, target = data.float().to(self.device), target.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                train_acc += (pred == target).sum().item()

                train_loader_iter.set_postfix({"Loss": loss.item()})

            train_loss /= len(train_loader)
            train_acc = train_acc / len(train_loader.dataset)

            # eval()
            self.model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.float().to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_acc += pred.eq(target.view_as(pred)).sum().item()
                    val_loss += criterion(output, target).item()

            val_loss /= len(val_loader)
            val_acc = val_acc / len(val_loader.dataset)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.valid_losses.append(val_loss)
            self.valid_accs.append(val_acc)

            print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss:.4f}, "
                  f"Val loss: {val_loss:.4f}, Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}")

            if val_acc > best_val_acc:
                torch.save(self.model.state_dict(), "./weight/densenet121_best.pt")
                best_val_acc = val_acc

            # save the model state and optimizer state after each epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'val_losses': self.valid_losses,
                'val_accs': self.valid_accs,
            }, "./weight/densenet121_checkpoint.pt")
            torch.save(self.model.state_dict(), f"./weight/densenet121_epoch{epoch}.pt")

        torch.save(self.model.state_dict(), "./weight/densenet121_last.pt")

        self.save_results_to_csv()
        self.plot_loss()
        self.plot_accuracy()

    def save_results_to_csv(self):
        df = pd.DataFrame({
            'Train Loss': self.train_losses,
            'Train ACC': self.train_accs,
            'Validation Loss': self.valid_losses,
            'Validation ACC': self.valid_accs
        })
        df.to_csv('./weight/train_val_result.csv', index=False)

    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_losses, label="Train loss")
        plt.plot(self.valid_losses, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("./weight/loss_plot.jpg")

    def plot_accuracy(self):
        plt.figure()
        plt.plot(self.train_accs, label="Train Accuracy")
        plt.plot(self.valid_accs, label="Valid Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("./weight/accuracy_plot.jpg")

    def run(self, args):
        # 클래스 수 설정
        out_features = 22

        self.model = densenet121(pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, out_features)
        self.model.to(self.device)

        train_transforms = A.Compose([
            A.Resize(width=640, height=640),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),  # RandomBrightness 대신 RandomBrightnessContrast 사용
            A.RandomRotate90(),
            A.RandomGamma(),
            ToTensorV2()
        ])

        val_transforms = A.Compose([
            A.Resize(width=640, height=640),
            ToTensorV2()
        ])

        # dataset and dataloader
        train_dataset = CustomDataset(args.train_dir, transform=train_transforms)
        val_dataset = CustomDataset(args.val_dir, transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        epochs = args.epochs
        criterion = CrossEntropyLoss().to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=args.learning_rate,
                          weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        start_epoch = 0

        if args.resume_training:
            checkpoint = torch.load(args.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.train_accs = checkpoint['train_accs']
            self.valid_losses = checkpoint['val_losses']
            self.valid_accs = checkpoint['val_accs']
            start_epoch = checkpoint['epoch']

        self.train(train_loader, val_loader, epochs, optimizer, criterion, scheduler, start_epoch=start_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./dataset/train/",
                        help='directory path to the training dataset')
    parser.add_argument("--val_dir", type=str, default="./dataset/valid/",
                        help='directory path to the valid dataset')
    parser.add_argument("--epochs", type=int, default=25,
                        help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=14,
                        help="batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay for optimizer")
    parser.add_argument("--resume_training", action='store_true',
                        help="resume training from the last checkpoint")
    parser.add_argument("--checkpoint_path", type=str,
                        default="./weight/densenet121_checkpoint.pt",
                        help="path to the checkpoint file")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default="./weight")

    args = parser.parse_args()

    weight_folder_path = args.checkpoint_folder_path
    os.makedirs(weight_folder_path, exist_ok=True)

    classifier = Fashion_Recommendation()
    classifier.run(args)
