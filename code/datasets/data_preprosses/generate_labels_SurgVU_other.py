import numpy as np
import os
import pickle
from tqdm import tqdm

def main():
    ROOT_DIR = "/jhcnas4/syangcw/surgvu24/"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'frames'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])

    TRAIN_NUMBERS = np.arange(0,135).tolist()
    TEST_NUMBERS = np.arange(135,155).tolist()

    TRAIN_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    train_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_test = 0

    phase2id = {"Suturing": 5, "Uterine horn": 6, "Suspensory ligaments": 4, 
                "Rectal artery/vein": 1, "Skills application": 3, "Range of motion": 0, 
                "Retraction and collision avoidance": 2, "other": 7}

    for video_id in VIDEO_NAMES:
        print(video_id)
        vid_id = int(video_id.split('_')[1])
        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test

        # 总帧数(frames)
        video_path = os.path.join(ROOT_DIR, "frames", video_id)
        frames_list = os.listdir(video_path)

        # 打开Label文件
        phase_path = os.path.join(ROOT_DIR, 'labels', video_id + '.txt')
        phase_results = open(phase_path, 'r')
        index_phase = dict()
        for phase_anno in phase_results:
            phase_annos = phase_anno.split(";")
            index, label = phase_annos[0], phase_annos[1].strip()
            index_phase[int(index)] = label

        frame_infos = list()
        for frame_id in tqdm(range(0, len(frames_list))):
            info = dict()
            info['unique_id'] = unique_id
            phase = index_phase[frame_id]
            info['frame_id'] = frame_id
            info['video_id'] = video_id
            info['frames'] = len(frames_list)
            phase_id = int(phase2id[phase])
            info['phase_gt'] = phase_id
            info['phase_name'] = phase
            frame_infos.append(info)
            unique_id += 1
        print(len(frame_infos), len(frames_list))
        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += len(frames_list)
            unique_id_train = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += len(frames_list)
            unique_id_test = unique_id
    
    train_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)

    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 

if __name__ == '__main__':
    main()