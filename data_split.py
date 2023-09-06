import os
import shutil
from sklearn.model_selection import train_test_split

data_root = './dataset/both'  # 데이터 루트 디렉토리
train_root = os.path.join(data_root, 'train')  # 훈련 데이터 저장 디렉토리
valid_root = os.path.join(data_root, 'valid')  # 검증 데이터 저장 디렉토리

# 클래스(라벨) 리스트
classes = ['blouse', 'cardigan', 'cargo_pants', 'cotton_pants', 'denim_pants', 'hooded', 'leggings', 'long-sleeved_T-shirt',
           'onepiece', 'shirt', 'skirt', 'slacks', 'sport_pants', 'sweatshirt', 'zipup']  # 클래스 이름들

# 클래스별 이미지 경로 리스트 생성
image_paths_by_class = {class_name: [] for class_name in classes}

# 클래스별 이미지 경로 수집
for class_name in classes:
    class_dir = os.path.join(data_root, class_name)  # 클래스 폴더 경로 수정
    image_paths = [os.path.join(class_dir, img) for img in os.listdir(class_dir)]
    image_paths_by_class[class_name] = image_paths

# 훈련 및 검증 데이터셋으로 나누기
train_images = []
valid_images = []
train_labels = []
valid_labels = []

for class_name, image_paths in image_paths_by_class.items():
    class_train_images, class_valid_images = train_test_split(image_paths, test_size=0.1, random_state=42)
    train_images.extend(class_train_images)
    valid_images.extend(class_valid_images)
    train_labels.extend([class_name] * len(class_train_images))
    valid_labels.extend([class_name] * len(class_valid_images))

# 데이터 복사 및 폴더 구조 유지
def copy_images_to_directory(image_paths, labels, target_directory):
    for img, label in zip(image_paths, labels):
        dest_dir = os.path.join(target_directory, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(img))
        shutil.copy(img, dest_path)

# 훈련 데이터 복사
copy_images_to_directory(train_images, train_labels, train_root)

# 검증 데이터 복사
copy_images_to_directory(valid_images, valid_labels, valid_root)
