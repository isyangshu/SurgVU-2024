export CC=gcc-11
export CXX=g++-11

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 12324 \
training.py \
--batch_size 6 \
--epochs 10 \
--save_ckpt_freq 2 \
--model  surgformer_HTA \
--pretrained_path pretrain_params/timesformer_base_patch16_224_K400.pyth \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 2 \
--data_path /jhcnas4/syangcw/surgvu24 \
--eval_data_path /jhcnas4/syangcw/surgvu24 \
--nb_classes 8 \
--data_strategy online \
--output_mode key_frame \
--num_frames 24 \
--sampling_rate 4 \
--data_set SurgVU \
--data_fps 1fps \
--output_dir results/ \
--log_dir results/ \
--num_workers 10 \
--enable_deepspeed \
--dist_eval \
--no_auto_resume