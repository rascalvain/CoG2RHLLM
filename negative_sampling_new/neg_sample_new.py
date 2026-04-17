import json
import random


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def create_negative_samples(data):
    negative_samples = []
    knowledge_bases = [item['knowledge_base'] for item in data]

    for item in data:
        current_kb = item['knowledge_base']
        # 过滤出与当前知识库不同的知识库
        different_kbs = [kb for kb in knowledge_bases if kb != current_kb]

        if different_kbs:
            # 随机选择一个不同的知识库
            negative_kb = random.choice(different_kbs)
            negative_sample = {
                "history": item['history'],
                "response": item['response'],
                "knowledge_base": negative_kb,
                "dialogue_id": item['dialogue_id'],
                "type": 0  # 将负样本的 type 设置为 0
            }
            negative_samples.append((item, negative_sample))  # 保存正样本和负样本的元组

    return negative_samples


def save_data(samples, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for positive_sample, negative_sample in samples:
            f.write(json.dumps(positive_sample) + '\n')  # 写入正样本
            f.write(json.dumps(negative_sample) + '\n')  # 写入负样本


def main():
    input_file = './positive_sample.txt'  # 输入文件名
    output_file = './samples_with_negatives.txt'  # 输出文件名

    # 加载数据
    data = load_data(input_file)

    # 创建负样本
    negative_samples = create_negative_samples(data)

    # 保存正样本和负样本
    save_data(negative_samples, output_file)


if __name__ == "__main__":
    main()
