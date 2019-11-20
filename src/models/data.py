import os
import glob
import math

from onmt.utils.logging import logger
import torch.utils.data as data
import torch


class FileDataset(torch.utils.data.IterableDataset):

    def __init__(self, pt_file, use_interval=True, testing=False):
        super(FileDataset).__init__()
        self.use_interval = use_interval
        self.testing = testing
        self.pt_file = pt_file

    def __iter__(self):
        dataset = torch.load(self.pt_file)
        logger.info('Loading dataset from %s, number of examples: %d' %
                    (self.pt_file, len(dataset)))
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = len(dataset)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(dataset))

        for pos, ex in enumerate(dataset):
            # Constrain to worker's portion and ignore empty examples
            if pos < iter_start or not len(ex['src']): continue
            if pos > iter_end: return

            ex = self._parse_example(ex)
            if ex is None: continue
            yield ex

    def _parse_example(self, ex):
        """ Parses a single example read from .pt file """
        word_ids = ex['src']
        labels = ex['labels'] if 'labels' in ex else ex['src_sent_labels']

        segment_ids = ex['segs']
        if not self.use_interval:
            segment_ids = [0] * len(segment_ids)

        cls_indices = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        if self.testing:
            return word_ids, labels, segment_ids, cls_indices, src_txt, tgt_txt
        else:
            return word_ids, labels, segment_ids, cls_indices

    @staticmethod
    def batch(ex_list, testing):
        """ Batches a list of training examples """
        # Get batch size from length of example list
        size = len(ex_list)

        # Slice out raw features from list of example tuples
        pre_word_ids = [x[0] for x in ex_list]
        pre_labels = [x[1] for x in ex_list]
        pre_segment_ids = [x[2] for x in ex_list]
        pre_cls_indices = [x[3] for x in ex_list]

        # Pad and mask, ensure common dims across minibatch
        word_ids = torch.tensor(FileDataset._pad(pre_word_ids, 0))
        labels = torch.tensor(FileDataset._pad(pre_labels, 0))
        segment_ids = torch.tensor(FileDataset._pad(pre_segment_ids, 0))
        cls_indices = torch.tensor(FileDataset._pad(pre_cls_indices, -1))

        # Generate masks based on above tensors
        mask_cls = ~ (cls_indices == -1)
        cls_indices[cls_indices == -1] = 0
        mask = ~ (word_ids == 0)

        # Add str features if testing
        if testing:
            src_str = [x[-2] for x in ex_list]
            tgt_str = [x[-1] for x in ex_list]
            data = [word_ids, mask, labels, segment_ids, cls_indices, mask_cls, src_str, tgt_str]
        else:
            data = [word_ids, mask, labels, segment_ids, cls_indices, mask_cls]

        return data, size


    @staticmethod
    def _pad(data, value, width=None):
        # If width not given, autoselect based on longest subseq
        if width == None:
            width = max(len(d) for d in data)
        return [d + [value] * (width - len(d)) for d in data]


def load(args, corpus_type, device):
    logger.info("Loading %s data", corpus_type)
    assert corpus_type in ["train", "valid", "test"]
    testing = corpus_type == 'test'

    # Get list of input files for given corpus type
    pattern = '*%s.[0-9]*.pt' % corpus_type
    full_glob = args.src_path + pattern
    pts = sorted(glob.glob(full_glob))

    # Assemble ChainDataset from file level datasets
    datasets = [FileDataset(f, args.use_interval, testing) for f in pts]
    dataset = torch.utils.data.ChainDataset(datasets)

    return torch.utils.data.DataLoader(
            dataset,
            collate_fn=lambda x : FileDataset.batch(x, testing),
            drop_last=True,
            pin_memory=True
    )
