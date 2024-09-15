# SurgVU-2024

## Code
The code for training and testing.
```bash
code/
    └──dataset/
    ├──model/
    ├──datasets_phase.py
    ├──engine_for_phase.py
    ├──train.sh
    ├──test.sh
    ├──training.py
    └──utils.py
```

### Dataset

```shell
# Extract frames form raw videos
cd code/datasets/data_preprosses/
python extract_frames_SurgVU.py

# transfer labels form raw csv files to txt files
python extract_labels_SurgVU.py

# Generate .pkl for training
python generate_labels_SurgVU_other.py

# Note that you can change the size of each frame to reduce I/O load.
```
> SurgVU dataset code in ``code/datasets/phase/SurgVU_phase.py``

### Training
### Pretrained Parameters

We use the parameters of [TimeSformer](https://github.com/facebookresearch/TimeSformer) trained on [K400 dataset](https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0) with frames 8 and spatial size 224 as initialization parameters.

We provide the script for training [train.sh](https://github.com/isyangshu/SurgVU-2024/blob/main/code/train.sh).

run the following code for training

```shell
sh train.sh
```

We use frame_length = 16 and sampling_rate = 4 for training. During the training process, we compute image-level accuracy on the validation set as a performance metric to comprehensively evaluate the model and select the best checkpoint based on the optimal value. When evaluating within the Docker container, we additionally use the mean weighted F1-score to select the inference hyperparameters, such as sequence length and sampling intervals.