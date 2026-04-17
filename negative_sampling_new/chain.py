import pandas as pd

# 读取CSV文件
df = pd.read_csv('val_filter_new.csv')


# 定义一个函数来构造思维链
def construct_thinking_chain(row):
    knowledge_base = row['knowledge_base']
    response = row['response']

    # 分割知识内容为三元组
    triples = knowledge_base.split('<triple>')

    # 初始化推理路径和推理步骤
    reasoning_path = []
    reasoning_steps = []

    # 遍历三元组，构造推理路径和推理步骤
    for triple in triples:
        elements = triple.split('<sep>')
        if len(elements) >= 3:
            A = elements[0].strip()
            r1 = elements[1].strip()
            B = elements[2].strip()
            reasoning_path.append(f"{A}->{r1}->{B}")
            reasoning_steps.append(f"Entity {A} is related to entity {B} through relation {r1}")

    # 构造最终的思维链
    reasoning_path_str = '->'.join(reasoning_path)
    conclusion = f"<answer>{response}</answer>"  # 使用 response 列的内容作为结论

    # 返回思维链，使用 <think> 和 </think> 标签包裹
    thinking_chain = (
            f"<think>\n"
            f"Reasoning path is {reasoning_path_str}\n" +
            "\n".join(reasoning_steps) + "\n" +"</think>"+"\n"+conclusion
    )

    return thinking_chain


# 应用函数并创建新列
df['thinking_chain'] = df.apply(construct_thinking_chain, axis=1)

# 保存修改后的数据集
df.to_csv('val_filter_chain.csv', index=False)

print("New column 'thinking_chain' has been successfully added to the dataset!")
