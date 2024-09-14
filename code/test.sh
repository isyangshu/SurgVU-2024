export CC=gcc-11
export CXX=g++-11

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port 12324 \
training.py \
--batch_size 4 \
--epochs 10 \
--save_ckpt_freq 2 \
--model surgformer_HTA \
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
--num_frames 16 \
--sampling_rate 4 \
--eval \
--finetune ../submission/checkpoint-best.pth \
--data_set SurgVU \
--data_fps 1fps \
--output_dir results/ \
--log_dir results/ \
--num_workers 10 \
--enable_deepspeed \
--dist_eval