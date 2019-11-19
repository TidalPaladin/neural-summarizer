import argparse
import os
from distutils.util import strtobool

def str2bool(v):
    """Parses bool args given to argparse"""
    try:
        return strtobool(v)
    except ValueError:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Read env vars for some defaults
SRC_DIR = os.environ['SRC_DIR']
ARTIFACT_DIR = os.environ['ARTIFACT_DIR']

train_file_prefix = 'cnndm'
default_data_path = os.path.join(SRC_DIR, train_file_prefix)
default_result_path = os.path.join(ARTIFACT_DIR, 'results')
default_temp_path = os.path.join(ARTIFACT_DIR, 'temp')
default_log_path = os.path.join(ARTIFACT_DIR, 'logs/cnndm.log')
default_model_path = os.path.join(ARTIFACT_DIR, 'models')

parser = argparse.ArgumentParser()

parser.add_argument("--encoder", default='baseline', type=str, choices=['baseline','placeholder'])
parser.add_argument("--mode", default='train', type=str, choices=['train','validate','test'])
parser.add_argument("--src_path", default=default_data_path)
parser.add_argument("--model_path", default=default_model_path)
parser.add_argument("--result_path", default=default_result_path)
parser.add_argument("--temp_dir", default=default_temp_path)
parser.add_argument("--bert_config_path", default='/app/bert_config_uncased_base.json')

parser.add_argument("--batch_size", default=1000, type=int)

parser.add_argument("--use_interval", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument("--ff_size", default=512, type=int)
parser.add_argument("--heads", default=4, type=int)
parser.add_argument("--inter_layers", default=2, type=int)

parser.add_argument("--param_init", default=0, type=float)
parser.add_argument("--param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--optim", default='adam', type=str)
parser.add_argument("--lr", default=1, type=float)
parser.add_argument("--beta1", default= 0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--decay_method", default='', type=str)
parser.add_argument("--warmup_steps", default=8000, type=int)
parser.add_argument("--max_grad_norm", default=0, type=float)

parser.add_argument("--save_checkpoint_steps", default=5, type=int)
parser.add_argument("--accum_count", default=1, type=int)
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--report_every", default=1, type=int)
parser.add_argument("--train_steps", default=1000, type=int)
parser.add_argument("--recall_eval", type=str2bool, nargs='?',const=True,default=False)

parser.add_argument('--visible_gpus', default='1', type=str)
parser.add_argument('--gpu_ranks', default='0', type=str)
parser.add_argument('--log_file', default=default_log_path)
parser.add_argument('--dataset', default='')
parser.add_argument('--seed', default=666, type=int)

parser.add_argument("--test_all", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("--test_from", default='')
parser.add_argument("--train_from", default='')
parser.add_argument("--report_rouge", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("--block_trigram", type=str2bool, nargs='?', const=True, default=True)
