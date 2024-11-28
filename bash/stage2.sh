# Example Param1
python stage2_adapt.py \
        --ckpt ./ckpts/CS_source.pth \
        --ckpts_path adapted_ckpt/adapted_result_1 \
        --data_root  /All_DATA/cityscapes \
        --dataset cityscapes \
        --domain_desc1 "driving under rain" \
        --domain_desc2 "driving at night" \
        --domain_desc3 "driving in snow" \
        --domain_desc4 "driving in fog" \
        --domain_desc5 "driving through fire" \
        --domain_desc6 "driving in sandstorm" \
        --lr 0.01 \
        --notes "RAW six" \
        --wandb_name RUN_NAME \
        --wandb_project PROJECT_NAME \
        --wandb_entity YOUR_ENTITY \
        --path_mu_sig ./save_100_param1/cs_six_rain \
        --path_mu_sig2 ./save_100_param1/cs_six_night \
        --path_mu_sig3 ./save_100_param1/cs_six_snow \
        --path_mu_sig4 ./save_100_param1/cs_six_fog \
        --path_mu_sig5 ./save_100_param1/cs_six_fire \
        --path_mu_sig6 ./save_100_param1/cs_six_sandstorm \
        --proj_lr 0.001 \
        --scale  0.01 \
        --total_itrs 2000 \
        --test_root /All_DATA/acdc \
        --testset ACDC \
        --freeze_BB \
        --train_aug


# Example Param2

# python stage2_adapt.py \
#         --ckpt ./ckpts/CS_source.pth \
#         --ckpts_path adapted_ckpt/adapted_result_2 \
#         --data_root  /All_DATA/cityscapes \
#         --dataset cityscapes \
#         --domain_desc1 "driving under rain" \
#         --domain_desc2 "driving at night" \
#         --domain_desc3 "driving in snow" \
#         --domain_desc4 "driving in fog" \
#         --domain_desc5 "driving through fire" \
#         --domain_desc6 "driving in sandstorm" \
#         --lr 0.01 \
#         --notes "RAW six" \
#         --wandb_name RUN_NAME \
#         --wandb_project PROJECT_NAME \
#         --wandb_entity YOUR_ENTITY \
#         --path_mu_sig ./save_100_param2/cs_six_rain \
#         --path_mu_sig2 ./save_100_param2/cs_six_night \
#         --path_mu_sig3 ./save_100_param2/cs_six_snow \
#         --path_mu_sig4 ./save_100_param2/cs_six_fog \
#         --path_mu_sig5 ./save_100_param2/cs_six_fire \
#         --path_mu_sig6 ./save_100_param2/cs_six_sandstorm \
#         --proj_lr 0.001 \
#         --scale  0.01 \
#         --total_itrs 2000 \
#         --test_root /All_DATA/acdc \
#         --testset ACDC \
#         --freeze_BB \
#         --train_aug

