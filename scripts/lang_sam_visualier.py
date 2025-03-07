import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def visualize_results(image_path, results, output_dir="vis_output", output_text_path="output_results.txt"):
    """
    객체 감지 및 세그멘테이션 결과를 시각화하고, 바이너리 마스크를 생성하는 함수

    Args:
        image_path (str): 원본 이미지 파일 경로
        results (list): 모델이 반환한 예측 결과 리스트 (각 항목은 dict)
        output_dir (str): 결과 파일을 저장할 디렉토리 (기본값: "output")

    Returns:
        None (이미지 및 마스크 파일 저장)
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 결과 이미지 저장 경로
    output_image_path = os.path.join(output_dir, "output_segmented.jpg")
    output_masked_image_path = os.path.join(output_dir, "output_masked.jpg")
    output_text_path = os.path.join(output_dir, "output_results.txt")
    output_merged_mask_path = os.path.join(output_dir, "output_mask_merged.png")

    # 원본 이미지 불러오기 (PIL -> OpenCV 변환)
    image_pil = Image.open(image_path)
    image = np.array(image_pil)  # PIL -> NumPy 배열 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV 형식(BGR)로 변환

    # 원본 이미지 복사본 (마스크 적용된 이미지 생성용)
    masked_image = image.copy()

    # 원본 크기의 빈 바이너리 마스크 (모든 객체 마스크를 병합할 용도)
    merged_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # 결과 시각화 및 바이너리 마스크 저장
    for idx, result in enumerate(results):
        boxes = result["boxes"]
        labels = result["labels"]
        scores = result["scores"]
        masks = result["masks"]

        for i in range(len(labels)):
            label = labels[i]
            score = scores[i]
            box = boxes[i].astype(int)  # (x1, y1, x2, y2)

            # 바운딩 박스 그리기
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # 객체 정보 텍스트 추가
            text = f"{label}: {score:.2f}"
            cv2.putText(image, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 세그멘테이션 마스크 적용
            mask = masks[i]

            # 바이너리 마스크 저장 (0과 255로 변환)
            binary_mask = (mask * 255).astype(np.uint8)
            mask_output_path = os.path.join(output_dir, f"output_mask_{idx}_{i}.png")
            cv2.imwrite(mask_output_path, binary_mask)
            print(f"🖼️ 바이너리 마스크 저장 완료: {mask_output_path}")

            # 객체 마스크를 병합 마스크에 추가 (논리 OR 연산)
            merged_mask = np.maximum(merged_mask, binary_mask)

            # 컬러 마스크 생성 (초록색 마스크 적용)
            color_mask = np.zeros_like(image, dtype=np.uint8)
            color_mask[:, :, 1] = 255  # 초록색 (BGR 형식)

            # 마스크를 원본 이미지 위에 반투명하게 덮어씌우기
            mask_indices = mask > 0
            overlay = image.copy()
            overlay[mask_indices] = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)[mask_indices]
            image = overlay

            # 마스크를 적용한 원본 이미지에도 반영
            masked_image[mask_indices] = color_mask[mask_indices]


    # 병합된 마스크 저장
    cv2.imwrite(output_merged_mask_path, merged_mask)
    print(f"🖼️ 모든 객체의 병합된 마스크 저장 완료: {output_merged_mask_path}")


    # 결과 이미지 저장
    cv2.imwrite(output_image_path, image)
    print(f"✅ 세그멘테이션 이미지 저장 완료: {output_image_path}")

    # 마스크 적용된 원본 이미지 저장
    cv2.imwrite(output_masked_image_path, masked_image)
    print(f"🖼️ 마스크 적용된 원본 이미지 저장 완료: {output_masked_image_path}")

    # 결과 정보 저장
    with open(output_text_path, "w") as f:
        for result in results:
            f.write(str(result) + "\n")

    print(f"📄 결과 정보 저장 완료: {output_text_path}")

    # 결과 이미지 표시 (Matplotlib)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Segmented Result")
    plt.show()

    # 병합된 마스크 이미지 표시
    plt.figure(figsize=(8, 6))
    plt.imshow(merged_mask, cmap="gray")
    plt.axis("off")
    plt.title("Merged Binary Mask")
    plt.show()