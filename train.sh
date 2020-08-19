python multigpu_train.py \
--gpu_list=0,1 \
--input_size=512 \
--batch_size_per_gpu=16 \
--checkpoint_path=/home/xhyang/workspace/EAST-tf/ckpts/ \
--text_scale=800 \
--max_steps=100000 \
--training_gt_path=/home/xhyang/workspace/EAST-tf/train_data/gts \
--training_data_path=/home/xhyang/workspace/EAST-tf/train_data/images \
--geometry=RBOX \
--learning_rate=0.0001 \
--num_readers=24 \
--restore=False \
--pretrained_model_path=/home/xhyang/workspace/EAST-tf/pretrained_model/resnet_v1_50.ckpt

