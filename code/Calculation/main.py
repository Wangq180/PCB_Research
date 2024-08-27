import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
from PIL import Image
from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import Dataset

class SuperResolutionDataset(Dataset):
    def __init__(self, dataset_folders, hr_transform=None, lr_transform=None):
        self.hr_images = []
        self.lr_images = []
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.filenames = []
        for folder in dataset_folders:
            hr_dir = os.path.join(folder, 'sharp')
            lr_dir = os.path.join(folder, 'blur')

            # 这里假设高分辨率和低分辨率图片的文件名是匹配的
            filenames = sorted(os.listdir(lr_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
            filenames = [f for f in filenames if os.path.isfile(os.path.join(hr_dir, f))]

            for filename in filenames:
                self.hr_images.append(os.path.join(hr_dir, filename))
                self.lr_images.append(os.path.join(lr_dir, filename))
                self.filenames.append(filename)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_images[idx]).convert('RGB')
        lr_image = Image.open(self.lr_images[idx]).convert('RGB')

        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)
        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)

        return lr_image, hr_image

# Conversion of high-resolution and low-resolution images
hr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

lr_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])





import torch
import torch.nn as nn

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual

# EDSR Model, use scale_factor to choose scale
class EDSR(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, num_residual_blocks=16):
        super(EDSR, self).__init__()
        self.num_channels = num_channels

        # First layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)

        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=scale_factor),
            nn.ReLU(inplace=True)
        )

        # Output layer
        self.conv3 = nn.Conv2d(64, num_channels, kernel_size=9, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = out + residual  # Element-wise sum
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

# 定义主函数
def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 实例化模型并加载预训练权重
    model_path = 'venv/SuperResolution_EDSR.pth'
    model = EDSR(scale_factor=2, num_channels=3, num_residual_blocks=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 图像转换函数
    hr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    lr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集路径列表
    dataset_base = 'venv/data_mask'
    dataset_folders = [os.path.join(dataset_base, d) for d in os.listdir(dataset_base)
                       if os.path.isdir(os.path.join(dataset_base, d))]

    # 创建数据集和数据加载器
    val_dataset = SuperResolutionDataset(dataset_folders, hr_transform=hr_transform, lr_transform=lr_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 创建输出目录
    output_dir = 'venv/result'
    os.makedirs(output_dir, exist_ok=True)
    from torchvision.utils import save_image
    total_images = len(val_loader)
    with torch.no_grad():
        for idx, (lr_image, _) in enumerate(val_loader):
            lr_image = lr_image.to(device)
            sr_image = model(lr_image)
            sr_image = inv_normalize(sr_image.squeeze(0)).clamp(0, 1)
            filename = val_dataset.filenames[idx]
            save_image(sr_image, os.path.join(output_dir, filename))
            print(f"Processed {filename} ({idx + 1}/{total_images})")

    print("All images have been processed and saved.")


print("All images have been processed and saved.")

# 检查是否直接运行此脚本，并在是的情况下调用 main()
if __name__ == "__main__":
    main()