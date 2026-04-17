import csv

def split_txt_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
        # 分割数据块
        groups = content.split('</answer>')
        # 准备写入CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
        # 写入表头
            csv_writer.writerow(['think', 'answer'])
            for group in groups:
                if '<think>' in group and '<answer>' in group:
            # 提取think和answer内容
                    think_start = group.find('<think>') + len('<think>')
                    think_end = group.find('</think>')
                    think_content = group[think_start:think_end].strip()
                    answer_start = group.find('<answer>') + len('<answer>')
                    answer_content = group[answer_start:].strip()
                # 直接取到末尾
                # 写入CSV文件
                    csv_writer.writerow([think_content, answer_content])

# 使用示例
input_file = './B4_generated_predictions.txt'  # 输入文件名
output_file = 'output.csv'  # 输出文件名
split_txt_to_csv(input_file, output_file)
