#!/bin/sh
docker-compose run bertsum  \
		--mode test  \
		--test_from /artifacts/models/model_step_50000.pt  \
		--encoder bertsum  \
		--visible_gpus 1  \
		--gpu_ranks 0  \
		--batch_size 4
