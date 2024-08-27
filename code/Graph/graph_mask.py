import matplotlib.pyplot as plt

def parse_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    dcnn_psnr, edsrn_psnr, resnet_psnr = [], [], []
    dcnn_mse, edsrn_mse, resnet_mse = [], [], []

    for line in lines:
        try:
            parts = line.split(': ')[1].split(' - ')
            if len(parts) != 3:
                continue  # Skip lines that do not have complete data for all three models

            dcnn_psnr.append(float(parts[0].split(', MSE = ')[0].split('PSNR = ')[1].strip()))
            dcnn_mse.append(float(parts[0].split(', MSE = ')[1].strip()))

            edsrn_psnr.append(float(parts[1].split(', MSE = ')[0].split('PSNR = ')[1].strip()))
            edsrn_mse.append(float(parts[1].split(', MSE = ')[1].strip()))

            resnet_psnr.append(float(parts[2].split(', MSE = ')[0].split('PSNR = ')[1].strip()))
            resnet_mse.append(float(parts[2].split(', MSE = ')[1].strip()))
        except IndexError:
            print(f"Error processing line: {line.strip()}")
            continue

    return dcnn_psnr, edsrn_psnr, resnet_psnr, dcnn_mse, edsrn_mse, resnet_mse

def plot_data(psnr_list, mse_list, title, save_path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    for data, label in zip(psnr_list, ["DCNN", "EDSRN", "ResNet"]):
        plt.plot(data, label=label)
    plt.title('PSNR Comparison')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    plt.subplot(1, 2, 2)
    for data, label in zip(mse_list, ["DCNN", "EDSRN", "ResNet"]):
        plt.plot(data, label=label)
    plt.title('MSE Comparison')
    plt.xlabel('Image Index')
    plt.ylabel('MSE')
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    filename = 'merged_model_data.txt'  # 更正文件名
    save_path = 'model_performance_original.png'  # 指定保存图像的路径
    dcnn_psnr, edsrn_psnr, resnet_psnr, dcnn_mse, edsrn_mse, resnet_mse = parse_data(filename)
    plot_data([dcnn_psnr, edsrn_psnr, resnet_psnr], [dcnn_mse, edsrn_mse, resnet_mse], 'Mask original', save_path)

if __name__ == "__main__":
    main()
