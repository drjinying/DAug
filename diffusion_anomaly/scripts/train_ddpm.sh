OMP_NUM_THREADS=5 torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/image_train.py \
    --data_dir path_to_traindata \
    --dataset mimic-cxr \
    --image_size 512 \
    --num_channels 128 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --lr 1e-4 \
    --batch_size 2

python scripts/classifier_sample_known.py \
    --data_dir path_to_testdata \
    --model_path /yjblob/medical/ckpt/diff/mimic2update000011.pt \
    --classifier_path /yjblob/medical/ckpt/diff_classifier_multi/ep_014.pt \
    --dataset mimic-cxr \
    --classifier_scale 100 \
    --noise_level 500 \
    --image_size 512 \
    --num_channels 128 \
    --class_cond True \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --classifier_attention_resolutions 32,16,8 \
    --classifier_depth 4 \
    --classifier_width 32 \
    --classifier_pool attention \
    --classifier_resblock_updown True \
    --classifier_use_scale_shift_norm True \
    --batch_size 1 \
    --num_samples 1 \
    --timestep_respacing ddim1000 \
    --use_ddim True \
    --i_cls 0