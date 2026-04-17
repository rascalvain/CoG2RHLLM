import pandas as pd

# 读取CSV文件
df = pd.read_csv('D:/模型/大模型幻觉/model/RHO-main/RHO-main/deepseek_test/test.csv')


# 定义一个函数来提取知识内容
def extract_knowledge(history):
    # 查找 "Given the knowledge:" 的位置
    start = history.find("Given the knowledge:")
    if start != -1:
        # 提取知识内容的开始位置
        start += len("Given the knowledge:")

        # 查找下一个分隔符 <user> 或 <assistant>
        end_user = history.find("<user>", start)
        end_assistant = history.find("<assistant>", start)

        # 找到最小的结束位置
        end = len(history)  # 默认取到字符串末尾
        if end_user != -1:
            end = end_user
        if end_assistant != -1 and end_assistant < end:
            end = end_assistant

        # 提取知识内容并去除空格
        knowledge = history[start:end].strip()
        return knowledge
    return ''  # 如果没有找到，返回空字符串


# 应用函数并创建新列
df['knowledge_base'] = df['history'].apply(extract_knowledge)

# 保存修改后的数据集
df.to_csv('D:/模型/大模型幻觉/model/RHO-main/RHO-main/deepseek_test/test_new.csv', index=False)

print("新列 'knowledge_base' 已成功添加到数据集中！")
