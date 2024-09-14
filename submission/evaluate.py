import json
from sklearn.metrics import precision_score, recall_score, f1_score

# 读取json文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 计算准确率、召回率和F1
def calculate_metrics(true_json, pred_json):
    # 提取surgical_step字段
    true_values = [item['surgical_step'] for item in true_json]
    pred_values = [item['surgical_step'] for item in pred_json]
    
    # 类别为0到8
    labels = list(range(8))

    # 计算每个类别的精确率、召回率和F1得分
    precision = precision_score(true_values, pred_values, labels=labels, average=None)
    recall = recall_score(true_values, pred_values, labels=labels, average=None)
    f1 = f1_score(true_values, pred_values, labels=labels, average=None)

    # 输出每个类别的结果
    for i, label in enumerate(labels):
        print(f"Category {label}:")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall: {recall[i]:.4f}")
        print(f"F1-score: {f1[i]:.4f}")

    # 计算weighted mean F1-score
    mean_weighted_f1 = f1_score(true_values, pred_values, labels=labels, average='weighted')
    print(f"\nMean Weighted F1-score: {mean_weighted_f1:.4f}")

# 示例：读取真值和预测值的json文件
true_json = load_json('/home/syangcw/SurgVU/submission/output/case_145.json')  # 真值
pred_json = load_json('/home/syangcw/SurgVU/submission/output/surgical-step-classification.json')  # 预测值

# 计算并打印每个类别的精确度、召回率、F1得分以及mean weighted F1
calculate_metrics(true_json, pred_json)
