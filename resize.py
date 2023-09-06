import torch
from torchvision import transforms
from PIL import Image


# 이미지를 원본 비율을 유지하면서 가로 480, 세로 640 크기로 패딩하는 함수
def resize_with_padding(image, target_width, target_height):
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    # 원본 비율을 유지하면서 가로 480, 세로 640 크기로 조정
    if aspect_ratio > 1:
        new_width = int(target_width)
        new_height = int(target_width / aspect_ratio)
    else:
        new_width = int(target_height * aspect_ratio)
        new_height = int(target_height)

    # 크기를 조정하고 흰 배경색으로 패딩을 추가하여 목표 크기로 패딩
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    padded_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))  # 배경색을 흰색으로 지정
    left = (target_width - new_width) // 2
    top = (target_height - new_height) // 2
    padded_image.paste(resized_image, (left, top))

    return padded_image

# 이미지 로드 및 크기 조정
image_path = "your_image.jpg"  # 680x1000 크기의 이미지 파일 경로
image = Image.open(image_path)
target_width = 480
target_height = 640
resized_image = resize_with_padding(image, target_width, target_height)

# 변환된 이미지 저장 또는 활용
resized_image.save("resized_image.jpg")
