import os

folder_path = 'venv/result'  # 替换为图片文件夹的路径
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}  # 添加或删除以支持不同的图片格式

image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
image_count = len(image_files)

print(f'There are {image_count} image files in {folder_path}')