#!/bin/sh
docker-compose run bertsum --mode train --encoder baseline --dropout 0.1 --lr 2e-3 --visible_gpus 1  --gpu_ranks 0 --world_size 0 --report_every 50 --save_checkpoint_steps 1000 --batch_size 512 --decay_method noam --train_steps 50000 --accum_count 2 --heads 8 --use_interval true --warmup_steps 10000
