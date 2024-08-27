import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import matplotlib.pyplot as plt

class DeblurDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 遍历根目录下的所有子目录
        for subdir in sorted(os.listdir(root_dir)):
            blur_folder = os.path.join(root_dir, subdir, 'blur')
            sharp_folder = os.path.join(root_dir, subdir, 'sharp')
            # 假设每个文件夹中的文件名是对应的
            filenames = os.listdir(blur_folder)
            for filename in filenames:
                blur_path = os.path.join(blur_folder, filename)
                sharp_path = os.path.join(sharp_folder, filename)
                self.samples.append((blur_path, sharp_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        blur_path, sharp_path = self.samples[idx]
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')

        if self.transform:
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)

        return blur_img, sharp_img, os.path.basename(blur_path)


# Normalized
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = DeblurDataset(root_dir='venv/data_mask2', transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

import torch.nn as nn
import torch.nn.functional as F

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
# Definite U-Net
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Create U-Net model instance
model = UNet(n_channels=3, n_classes=3)
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.distributed as distance
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def Perceptual Loss by MobileNet
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.mobilenet.eval()
        for param in self.mobilenet.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.mobilenet(input)
        target_features = self.mobilenet(target)
        # Resize if necessary
        if input_features.shape[2:] != target_features.shape[2:]:
            input_features = F.interpolate(input_features, size=target_features.shape[2:], mode='bilinear',
                                           align_corners=False)

        loss = nn.functional.mse_loss(input_features, target_features)
        return loss


from torch.optim.lr_scheduler import ReduceLROnPlateau

# Creat model and Adam optimizer
debluring_model = UNet(n_channels=3, n_classes=3).to(device)
mse_criterion = nn.MSELoss()
perceptual_criterion = PerceptualLoss().to(device)
optimizer = torch.optim.Adam(debluring_model.parameters(), lr=0.0001)

# Initialize the ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

# Defining Loss Function Weights
mse_weight = 1.0
perceptual_weight = 0.1

# 1. Load the model and pretrained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=3)
model_path = 'venv/deblured_UNet.pth' # Update to the correct path
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 2. Create the DataLoader for your dataset
# 注意：您需要确保`DataLoader`中使用的是测试集
data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

# 3. Image inverse normalization transformation
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

# 4. Define the function to save deblurred images
output_folder = 'venv/data_resultmask2' # Update to the correct output path
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def save_deblurred_images(data_loader, output_folder):
    with torch.no_grad():
        for blur_img, _, filenames in data_loader:
            blur_img = blur_img.to(device)
            deblured_img = model(blur_img)
            deblured_img = inv_normalize(deblured_img)  # Apply inverse normalization
            deblured_img = deblured_img.clamp(0, 1)    # Clamp the values to the range [0, 1]

            for i in range(blur_img.size(0)):
                deblured_np = deblured_img[i].cpu().numpy().transpose(1, 2, 0)
                img_filename = filenames[i]
                plt.imsave(os.path.join(output_folder, img_filename), deblured_np)



# Save the images
if __name__ == '__main__':
    save_deblurred_images(data_loader, output_folder)