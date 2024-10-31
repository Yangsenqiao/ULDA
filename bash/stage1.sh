# Example Param1
python stage1_simulate.py \
    --total_it 100 \
    --temperature 10 \
    --save_dir1 save_100_param1/cs_six_rain \
    --save_dir2 save_100_param1/cs_six_night \
    --save_dir3 save_100_param1/cs_six_snow \
    --save_dir4 save_100_param1/cs_six_fog \
    --save_dir5 save_100_param1/cs_six_fire \
    --save_dir6 save_100_param1/cs_six_sandstorm \
    --region_rate 1.0 \
    --proj_lr 0.001 \
    --domain_rate 20 \
    --seg_rate 1 \
    --pixel_rate 0.5 \
    --cor_rate 0 \
    --batch_size 2 \
    --ckpt ./ckpts/CS_source.pth \
    --data_root /All_DATA/cityscapes \
    --dataset cityscapes \
    --domain_desc1 "driving under rain" \
    --domain_desc2 "driving at night" \
    --domain_desc3 "driving in snow" \
    --domain_desc4 "driving in fog" \
    --domain_desc5 "driving through fire" \
    --domain_desc6 "driving in sandstorm" \
    --notes 'RAW six ' \
    --wandb_name RUN_NAME \
    --wandb_project PROJECT_NAME \
    --wandb_entity YOUR_ENTITY \
    --resize_feat




# Example Param2

# python stage1_simulate.py \
#     --total_it 60 \
#     --temperature 10 \
#     --save_dir1 save_100_param2/cs_six_rain \
#     --save_dir2 save_100_param2/cs_six_night \
#     --save_dir3 save_100_param2/cs_six_snow \
#     --save_dir4 save_100_param2/cs_six_fog \
#     --save_dir5 save_100_param2/cs_six_fire \
#     --save_dir6 save_100_param2/cs_six_sandstorm \
#     --region_rate 0.5 \
#     --proj_lr 0.001 \
#     --domain_rate 10 \
#     --seg_rate 1 \
#     --pixel_rate 0.1 \
#     --cor_rate 10 \
#     --batch_size 2 \
#     --ckpt ./ckpts/CS_source.pth \
#     --data_root /All_DATA/cityscapes \
#     --dataset cityscapes \
#     --domain_desc1 "driving under rain" \
#     --domain_desc2 "driving at night" \
#     --domain_desc3 "driving in snow" \
#     --domain_desc4 "driving in fog" \
#     --domain_desc5 "driving through fire" \
#     --domain_desc6 "driving in sandstorm" \
#     --notes 'RAW six ' \
#     --wandb_name RUN_NAME \
#     --wandb_project PROJECT_NAME \
#     --wandb_entity YOUR_ENTITY \
#     --resize_feat




