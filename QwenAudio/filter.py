import json
import re
import json
import re

# 读取txt文件内容
with open('kwaibgm_part1_3.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
def process_description(description):
    # 去除'\'后面的所有内容
    cleaned_description = description.split("\n")[0]
    # 计算词数
    word_count = len(cleaned_description.split())
    return cleaned_description, word_count

def filter_descriptions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    filtered_data = []

    for line in lines:
        data = json.loads(line)
        description = data.get("caption", "")
        cleaned_description, word_count = process_description(description)
        if word_count >= 8 and word_count < 27:
            data["caption"] = cleaned_description
            filtered_data.append(data)

    # 将结果写回到一个新的文件中
    with open("kwaibgm_part1_3.txt", 'w', encoding='utf-8') as outfile:
        for item in filtered_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')

# 使用示例
file_path = "kwaibgm_part1_2.txt"  
filter_descriptions(file_path)



# 处理每一行
def clean_description(description):

    match = re.search(r'\b\w+[A-Z]\w*\b', description)
    if match:
        # 找到错误拼接的单词并进行纠错
        incorrect_word = match.group(0)
        if "mood" in incorrect_word:
            correct_word = "mood."
        elif "Someone" in incorrect_word:
            correct_word = incorrect_word.replace("Someone", "")
        elif "The" in incorrect_word:
            correct_word = incorrect_word.replace("The", "")
        elif "A" in incorrect_word:
            correct_word = incorrect_word.replace("A", "")
        elif "This" in incorrect_word:
            correct_word = incorrect_word.replace("This", "")
        else:
            correct_word = incorrect_word
        
        # 只保留不正确拼接单词前的内容并纠正
        index = match.start()
        return description[:index] + correct_word
    return description

output_lines = []


for line in lines:
    try:
        # 解析JSON
        data = json.loads(line.strip())
        description = data.get("caption", '')

        # 清理description
        # cleaned_description = clean_description(description)
        # data["caption"] = cleaned_description
        output_lines.append(data)
        # 生成新的JSON字符串
        # output_lines.append(json.dumps(data, ensure_ascii=False))
    except json.JSONDecodeError:
        continue

# 将结果写入新的txt文件
with open('Kwaimusic-Qwen.json', 'w', encoding='utf-8') as outfile:
    json.dump(output_lines, outfile, indent=4)


print("Processing completed. Check 'output.txt' for results.")

