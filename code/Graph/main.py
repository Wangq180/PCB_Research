


def read_model_data(file_name):
    """从指定的文件中读取数据"""
    with open(file_name, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

def merge_data(data1, data2, data3):
    """合并三个数据源的数据"""
    merged_data = []
    for d1, d2, d3 in zip(data1, data2, data3):
        filename = d1.split(': ')[0]
        psnr_mse1 = d1.strip().split(': ')[1]
        psnr_mse2 = d2.strip().split(': ')[1]
        psnr_mse3 = d3.strip().split(': ')[1]
        merged_line = f"{filename}: {psnr_mse1} - {psnr_mse2} - {psnr_mse3}"
        merged_data.append(merged_line)
    return merged_data

def save_merged_data(filename, data):
    """将合并后的数据保存到新文件中"""
    with open(filename, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line + '\n')

def main():
    # 读取数据
    data1 = read_model_data('DS_mask.txt')
    data2 = read_model_data('EDSRN_mask.txt')
    data3 = read_model_data('Res_mask.txt')

    # 合并数据
    merged_data = merge_data(data1, data2, data3)

    # 保存到新文件
    save_merged_data('merged_model_data.txt', merged_data)

if __name__ == '__main__':
    main()
