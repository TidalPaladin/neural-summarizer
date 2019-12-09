#!/usr/bin/env python
import os
import signal
import time

import torch
from pytorch_pretrained_bert import BertConfig

import distributed
from onmt.utils.logging import logger, init_logger

from flags import parser
from models import data
from models.model import load_model
from models.trainer import build_trainer
from others.utils import seed


def multi_main(func, args):
    """ Spawns 1 process per GPU """
    init_logger()
    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        proc_args = (func, args, device_id, error_queue)
        procs.append(mp.Process(target=run, args=proc_args, daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(func, args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size,
                                          args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        func(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""
    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener,
                                             daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def evaluate(args, device_id, checkpoint, step, mode):
    """ Evaluates a model in either validation or test mode """

    if mode not in ['valid', 'test']:
        raise ValueError('mode must be [test, valid], got %s' % mode)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    if checkpoint == '': checkpoint = args.test_from
    logger.info("Evaluating (%s) from checkpoint %s", mode, checkpoint)

    config = BertConfig.from_json_file(args.bert_config_path)
    model, _ = load_model(args, device, load_bert=False, checkpoint=checkpoint)
    model.eval()

    dataset = data.load(args, mode, device)
    trainer = build_trainer(args, device_id, model, None)

    if mode == 'valid':
        stats = trainer.validate(dataset, step)
    else:
        stats = trainer.test(dataset, step)

    return stats


def train(args, device_id):
    """ Starts training pipeline given CLI args """
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    seed(args.seed, device_id)

    def train_iter_fct():
        return data.load(args, 'train', device)

    if args.train_from != '':
        logger.info("Training from checkpoint %s", args.train_from)
        model, optim = load_model(args,
                                  device,
                                  load_bert=True,
                                  checkpoint=args.train_from)
    else:
        logger.info("Training without checkpoint")
        model, optim = load_model(args, device, load_bert=True)

    logger.info(model)
    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)


def main(args):
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.log_file = '%s_%s_%s' % (args.log_file, args.mode, current_time)
    init_logger(args.log_file)
    logger.info(args)

    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.world_size > 1):
        multi_main(train, args)
    elif (args.mode == 'train'):
        train(args, device_id)
    #elif (args.mode == 'validate'):
    elif (args.mode == 'test'):
        cp = args.test_from
        try:
            step = int(cp.split('.')[-2].split('_')[-1])
        except:
            step = 0
        evaluate(args, device_id, cp, step, mode='test')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
