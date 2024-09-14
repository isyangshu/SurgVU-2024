import json

# 映射关系
phase2id = {
    "Suturing": 5, 
    "Uterine horn": 6, 
    "Suspensory ligaments": 4,
    "Rectal artery/vein": 1, 
    "Skills application": 3, 
    "Range of motion": 0,
    "Retraction and collision avoidance": 2, 
    "other": 7
}

# 转换函数
def txt_to_json(input_txt, output_json):
    result = []
    
    # 打开并读取txt文件
    with open(input_txt, 'r') as file:
        lines = file.readlines()
        
        # 遍历每一行，拆分并映射
        for line in lines:
            frame_nr, step = line.strip().split(';')  # 拆分frame和surgical_step
            frame_nr = int(frame_nr)  # 将frame_nr转换为整数
            surgical_step = phase2id.get(step, -1)  # 从映射中获取surgical_step对应的值，默认值为-1表示未找到
            
            # 创建字典并添加到结果列表中
            result.append({"frame_nr": frame_nr, "surgical_step": surgical_step})
    
    # 将结果保存为json文件
    with open(output_json, 'w') as json_file:
        json.dump(result, json_file, indent=4)

# 使用示例
input_txt = '/jhcnas4/syangcw/surgvu24/labels/case_145.txt'  # 替换为你的txt文件路径
output_json = '/home/syangcw/SurgVU/submission/output/case_145.json'  # 输出json文件路径
txt_to_json(input_txt, output_json)
