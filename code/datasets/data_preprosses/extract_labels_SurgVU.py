import numpy as np
import os
import cv2
import csv
import math

# 8 tasks
ROOT_DIR = "/jhcnas4/syangcw/surgvu24/labels_"
VIDEO_DIR = "/jhcnas4/syangcw/surgvu24/videos"
FRAME_DIR = "/jhcnas4/syangcw/surgvu24/frames"
VIDEO_NAMES = os.listdir(ROOT_DIR)
VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'case' in x])
LABEL_DIR = "/jhcnas4/syangcw/surgvu24/labels"

if not os.path.exists(LABEL_DIR):
    os.makedirs(LABEL_DIR)

for video_name in VIDEO_NAMES:
    print(video_name)
    label_path = os.path.join(ROOT_DIR, video_name)
    task_label = os.path.join(label_path, "tasks.csv")
    video_path = os.path.join(VIDEO_DIR, video_name)
    video_files = list()
    for filename in os.listdir(video_path):
        if filename.endswith(".mp4"):
            video_files.append(os.path.join(video_path, filename))
    video_files = sorted(video_files)
    label_save_file = os.path.join(LABEL_DIR, video_name+'.txt')
    vidcap = cv2.VideoCapture(video_files[0])
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) // fps
    with open(task_label, 'r') as csvfile:
        # 创建 CSV 读取器
        label_reader = list()
        reader = csv.reader(csvfile)
        next(reader)
        # 逐行读取数据
        for row in reader:
            start_part, start_time, stop_part, stop_time, groundtruth_taskname = row[1], row[2], row[3], row[4], row[5]
        
            if start_part == stop_part and int(start_part) == 1:
                label_reader.append((math.ceil(float(start_time)), math.floor(float(stop_time)), groundtruth_taskname))
            elif start_part == stop_part and int(start_part) == 2:
                label_reader.append((video_length + math.ceil(float(start_time)), video_length + math.floor(float(stop_time)), groundtruth_taskname))
            else:
                label_reader.append((math.ceil(float(start_time)), video_length + math.floor(float(stop_time)), groundtruth_taskname))
    save_dict = dict()
    for label in label_reader:
        s, e, a = label
        for i in range(int(s), int(e)+1):  # For error
            save_dict[i] = a
    frame_path = os.path.join(FRAME_DIR, video_name)
    print(len(os.listdir(frame_path)))
    for i in range(len(os.listdir(frame_path))):
        if i not in save_dict.keys():
            save_dict[i] = 'other'
        elif save_dict[i] == '':
            save_dict[i] = 'other'
    
    sorted_dict = dict(sorted(save_dict.items()))
    with open(label_save_file, "w") as file:
        for i in sorted_dict.keys():
            file.write(str(i)+';'+sorted_dict[i]+'\n')