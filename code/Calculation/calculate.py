from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
import numpy as np
from PIL import Image
import os
from skimage import img_as_float32
from skimage.transform import resize

def ensure_same_size_and_type(image_a, image_b):
    # 确保图像为浮点类型
    image_a = img_as_float32(image_a)
    image_b = img_as_float32(image_b)

    # 调整尺寸以匹配
    if image_a.shape != image_b.shape:
        image_b = resize(image_b, image_a.shape[:2], anti_aliasing=True, preserve_range=True)
    return image_a, image_b
def calculate_metrics(original_image_path, processed_image_path):
    # 加载图像并转换为NumPy数组
    original_image = np.array(Image.open(original_image_path))
    processed_image = np.array(Image.open(processed_image_path))
    original_image, processed_image = ensure_same_size_and_type(original_image, processed_image)
    # 确保图像为浮点类型
    original_image = original_image.astype(np.float32)
    processed_image = processed_image.astype(np.float32)

    # 计算PSNR和MSE
    psnr_value = psnr(original_image, processed_image)
    mse_value = mse(original_image, processed_image)

    return psnr_value, mse_value

def main(original_images_dir, processed_images_dir):
    processed_image_files = os.listdir(processed_images_dir)

    total_psnr = 0.0
    total_mse = 0.0

    # 用于存储至少处理了一个图像的标志
    processed_flag = False

    for image_file in processed_image_files:
        original_image_path = os.path.join(original_images_dir, image_file)
        processed_image_path = os.path.join(processed_images_dir, image_file)

        if os.path.exists(original_image_path):
            psnr_value, mse_value = calculate_metrics(original_image_path, processed_image_path)
            print(f"{image_file}: PSNR = {psnr_value:.2f}, MSE = {mse_value:.2f}")
            total_psnr += psnr_value
            total_mse += mse_value
            processed_flag = True

    if processed_flag:
        average_psnr = total_psnr / len(processed_image_files)
        average_mse = total_mse / len(processed_image_files)
        print(f"Average PSNR: {average_psnr:.2f}, Average MSE: {average_mse:.2f}")
    else:
        print("No images were processed.")

# 指定原始图像和处理后图像的目录
original_images_dir = 'venv/打码原图'  # 更改为你的原始图像目录
processed_images_dir = 'venv/result'  # 更改为你的处理后图像目录

if __name__ == "__main__":
    main(original_images_dir, processed_images_dir)

