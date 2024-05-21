import json

def process_tags(tags):
    # 分割标签并去掉 '---' 以及之前的部分
    tag_list = tags.split('\t')
    processed_tags = [tag.split('---')[-1] for tag in tag_list]
    processed_tags = ','.join(processed_tags)
    return processed_tags

def txt_to_json(txt_file, json_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    
    # 获取标题行并去掉换行符
    headers = lines[0].strip().split('\t')
    
    # 初始化一个空列表来存储所有记录
    data = []
    
    # 处理每一行数据
    for line in lines[1:]:
        # 去掉换行符并按制表符分割
        values = line.strip().split('\t')
        
        # 创建一个字典来存储当前记录
        record = dict(zip(headers, values))
        
        # 处理TAG字段
        record['TAGS'] = process_tags('\t'.join(values[5:]))  # 合并多列标签并处理
        
        # 仅保留PATH, DURATION, TAGS字段
        filtered_record = {
            'location': "/mmu-audio-ssd/frontend/audioSep/wanghualei/code/mtg-jamendo-dataset/path/to/raw_30s/audiofile/"+record['PATH'],
            'duration': float(record['DURATION']),  # 将DURATION转换为浮点数
            'tags': record['TAGS']
        }
        
        # 将处理后的记录添加到数据列表中
        data.append(filtered_record)
    
    # 将数据写入JSON文件
    with open(json_file, 'w') as json_f:
        json.dump(data, json_f, indent=4)

# 文件路径
txt_file = 'json_files/raw_30s_cleantags.tsv'
json_file = 'json_files/raw_30s_cleantags.json'

# 调用函数进行转换
txt_to_json(txt_file, json_file)


json_path ="json_files/tag.json"
with open(json_path, 'r') as f:
    data = json.load(f)
    
result=[]
for item in data:
    item['tags'] = item['tags'].lower()
    result.append(item)

with open('tag.json', 'w') as f:
    json.dump(result, f, indent=4)