def read_data(file_name):
    """从文件中读取数据"""
    with open(file_name, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return data

def parse_and_sort(data):
    """解析数据，并根据平均PSNR排序"""
    parsed_data = []
    for line in data:
        try:
            parts = line.strip().split(': ')
            filename = parts[0]
            details = parts[1]
            psnr_parts = details.split(' - ')
            psnr_values = []
            for part in psnr_parts:
                psnr_value = part.split(', MSE = ')[0].split('PSNR = ')[1]
                psnr_values.append(float(psnr_value))
            avg_psnr = sum(psnr_values) / len(psnr_values)
            parsed_data.append((filename, details, avg_psnr))
        except IndexError:
            print(f"Skipping line due to parsing error: {line}")

    # 从高到低排序PSNR（简单到难）
    parsed_data.sort(key=lambda x: x[2], reverse=True)

    return [item[0] + ': ' + item[1] for item in parsed_data]

def save_sorted_data(filename, sorted_data):
    """将排序后的数据保存到新文件中"""
    with open(filename, 'w', encoding='utf-8') as file:
        for entry in sorted_data:
            file.write(entry + '\n')

def main():
    # 读取原始数据
    data = read_data('merged_model_data.txt')

    # 解析并排序
    sorted_data = parse_and_sort(data)

    # 保存到新文件
    save_sorted_data('sorted_data——mask.txt', sorted_data)

if __name__ == '__main__':
    main()
