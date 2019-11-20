#!/bin/sh
docker-compose run bertsum \
		--mode train \
		--encoder bertsum \
		--dropout 0.1 \
		--learning_rate 1e-3 \
		--visible_gpus 1 \
		--gpu_ranks 0 \
		--world_size 1 \
		--report_every 50 \
		--save_checkpoint_steps 1000 \
		--batch_size 1 \
		--decay_method noam \
		--train_steps 50000 \
		--accum_count 2 \
		--heads 8 \
		--use_interval true \
		--warmup_steps 10000
