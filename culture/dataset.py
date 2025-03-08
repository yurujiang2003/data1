import json
import random

def process_data(input_path, output_paths, test_ratio=0.1):
    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化数据结构
    format1 = {"train": {}, "test": {}}  # Background + Country
    format2 = {"train": {}, "test": {}}  # Background + Country + Value
    format3 = {"train": {}, "test": {}}  # Background + Rule-of-thumb
    
    for split in data:  # 例如 "train"
        # 获取所有ID并随机打乱
        all_ids = list(data[split].keys())
        random.shuffle(all_ids)
        
        # 计算测试集大小
        test_size = int(len(all_ids) * test_ratio)
        
        # 划分训练集和测试集ID
        test_ids = all_ids[:test_size]
        train_ids = all_ids[test_size:]
        
        # 处理训练集数据
        for id in train_ids:
            item = data[split][id]
            
            # 格式1: Background + Country
            format1["train"][id] = {
                "instruction": f"{item['Background']}\nCountry: {item['Country']}.\nPlease justify: {item['Story']}.\n Choose from:'Yes', 'No', 'Neutral'",
                "label": item['Gold Label']
            }
            
            # 格式2: Background + Country + Value
            format2["train"][id] = {
                "instruction": f"{item['Background']}\nCountry: {item['Country']}\nValue: {item['Value']}\nPlease justify: {item['Story']}.\n Choose from:'Yes', 'No', 'Neutral'",
                "label": item['Gold Label']
            }
            
            # 格式3: Background + Rule-of-thumb
            format3["train"][id] = {
                "instruction": f"{item['Background']}\nRule of Thumb: {item['Rule-of-Thumb']}\nPlease justify: {item['Story']}.\n Choose from:'Yes', 'No', 'Neutral'",
                "label": item['Gold Label']
            }
        
        # 处理测试集数据
        for id in test_ids:
            item = data[split][id]
            
            # 格式1: Background + Country
            format1["test"][id] = {
                "instruction": f"{item['Background']}\nCountry: {item['Country']}.\nPlease justify: {item['Story']}.\n Choose from:'Yes', 'No', 'Neutral'",
                "label": item['Gold Label']
            }
            
            # 格式2: Background + Country + Value
            format2["test"][id] = {
                "instruction": f"{item['Background']}\nCountry: {item['Country']}\nValue: {item['Value']}\nPlease justify: {item['Story']}.\n Choose from:'Yes', 'No', 'Neutral'",
                "label": item['Gold Label']
            }
            
            # 格式3: Background + Rule-of-thumb
            format3["test"][id] = {
                "instruction": f"{item['Background']}\nRule of Thumb: {item['Rule-of-Thumb']}\nPlease justify: {item['Story']}.\n Choose from:'Yes', 'No', 'Neutral'",
                "label": item['Gold Label']
            }
    
    # 保存三种格式的数据
    for path, data in zip(output_paths, [format1, format2, format3]):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# 设置随机种子以确保可重复性
random.seed(42)

# 使用示例
input_path = "sparta_alignment/data/culture/normad_dataset.json"
output_paths = [
    "sparta_alignment/data/culture/country_dataset.json",
    "sparta_alignment/data/culture/country_value_dataset.json",
    "sparta_alignment/data/culture/rule_of_thumb_dataset.json"
]
process_data(input_path, output_paths, test_ratio=0.1)