import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt
import os

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.models.densenet import densenet121
from custom_dataset import CustomDataset
from tqdm import tqdm

# 결과를 저장할 폴더 생성
result_folder = "result"
os.makedirs(result_folder, exist_ok=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 설정
    model = densenet121(pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 22)
    model.load_state_dict(torch.load(f="./weight/densenet121_best.pt", map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    # Augmentation 및 데이터 로더 설정
    val_transforms = A.Compose([
        A.Resize(height=640, width=640),
        ToTensorV2()
    ])
    test_dataset = CustomDataset("./dataset/test(WH)/", transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    correct = 0
    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    #         if pred == target.item():
    #             correct_samples.append((data, pred, target))
    #         else:
    #             incorrect_samples.append((data, pred, target))
    #
    #         # 예측 확률 분포 시각화
    #         prob_dist = F.softmax(output, dim=1)[0].cpu().numpy()
    #         plt.bar(range(22), prob_dist)
    #         plt.xticks(range(22))
    #         plt.xlabel('Class')
    #         plt.ylabel('Probability')
    #         plt.title(f"Predicted Class: {pred}, Actual Class: {target.item()}")
    #         plt.savefig(f"./result/prob_dist_{idx}.png")
    #         # plt.show()
    #
    #         # 이미지 및 예측 클래스 시각화
    #         image = data[0].cpu().permute(1, 2, 0).numpy()
    #         plt.imshow(image)
    #         plt.title(f"Predicted Class: {pred}, Actual Class: {target.item()}")
    #         plt.savefig(f"./result/result_{idx}.png")
    #         # plt.show()
    #
    # # 올바른 예측 시각화
    # for idx, (data, pred, target) in enumerate(correct_samples[:5]):
    #     image = data[0].cpu().permute(1, 2, 0).numpy()
    #     plt.imshow(image)
    #     plt.title(f"Correct Prediction: Predicted Class {pred}, Actual Class {target.item()}")
    #     plt.savefig(f"./result/correct_{idx}.png")
    #     plt.show()
    #
    # # 잘못된 예측 시각화
    # for idx, (data, pred, target) in enumerate(incorrect_samples[:5]):
    #     image = data[0].cpu().permute(1, 2, 0).numpy()
    #     plt.imshow(image)
    #     plt.title(f"Incorrect Prediction: Predicted Class {pred}, Actual Class {target.item()}")
    #     plt.savefig(f"./result/incorrect_{idx}.png")
    #     plt.show()

    print("test set : Acc {}/{} [{:.0f}]%\n".format(
        correct, len(test_loader.dataset),
        100 * correct / len(test_loader.dataset)
    ))


if __name__ == "__main__":
    main()
