import os
import random
import re
import shutil
import time

from onmt.utils.logging import logger
import pyrouge
import torch

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}

def seed(value, device_id):
    """Sets all random seed values"""
    torch.manual_seed(value)
    random.seed(value)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(value)


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def test_rouge(temp_dir, candidates, references, pool_id=None, rouge_path='/app/rouge'):
    candidates = [line.strip() for line in open(candidates, encoding='utf-8')]
    references = [line.strip() for line in open(references, encoding='utf-8')]
    assert len(candidates) == len(references)

    # Temp dirs for gold standard / candidate summaries
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    if not pool_id:
        tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    else:
        tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    cand_dir = os.path.join(tmp_dir, "candidate")
    ref_dir = os.path.join(tmp_dir, "reference")

    # Create all dirs
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(cand_dir)
        os.mkdir(ref_dir)

    try:
        # Write candidate / reference values to files for ROUGE
        for i, (cand, ref) in enumerate(zip(candidates, references)):
            if len(ref) < 1: continue

            cand_file = os.path.join(cand_dir, 'cand.{}.txt'.format(i))
            with open(cand_file, "w", encoding="utf-8") as f:
                f.write(cand)

            ref_file = os.path.join(ref_dir, 'ref.{}.txt'.format(i))
            with open(ref_file, "w", encoding="utf-8") as f:
                f.write(ref)

        # Initialize ROUGE
        r = pyrouge.Rouge155(rouge_path)
        r.config_file = os.path.join(tmp_dir, 'rouge_config.xml')
        r.model_dir = ref_dir
        r.system_dir = cand_dir
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'

        # Run ROUGE and process results
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
        logger.info(
            "Rouge results:\n%s\n%s",
            rouge_results,
            rouge_results_to_str(results_dict)
        )
    finally:
        # Temp ROUGE file cleanup
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    return results_dict

def rouge_results_to_str(results_dict):
    return "ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )
