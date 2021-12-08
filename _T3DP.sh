# # test posetrack
# python test_t3dp.py \
# --dataset          "posetrack" \
# --dataset_path     "_DATA/Posetrack_2018/" \
# --storage_folder   "Videos_Final" \
# --th_x             20000 \
# --past_x           20 \
# --max_age_x        20 \
# --n_init_x         5 \
# --max_ids_x        50 \
# --window_x         1 \
# --downsample       1 \
# --metric_x         "euclidean_min" \
# --render           False \
# --save             False

        
# # test mupots
# python test_t3dp.py \
# --dataset          "mupots" \
# --dataset_path     "/_DATA/MuPoTs/" \
# --storage_folder   "Videos_Final" \
# --th_x             20000 \
# --past_x           1000 \
# --max_age_x        1000 \
# --n_init_x         5 \
# --max_ids_x        10 \
# --window_x         10 \
# --downsample       1 \
# --metric_x         "euclidean_min" \
# --render           False \
# --save             False





        
# train t3dp transformer
python train_t3dp.py \
--learning_rate      0.001 \
--lr_decay_epochs    10000,20000 \
--epochs             100000 \
--tags               T3DP \
--train_dataset      posetrack_2018 \
--test_dataset       posetrack_2018 \
--train_batch_size   32 \
--feature            APK \
--train







