import torch
from tqdm import tqdm
import logging

from codes.Dataset.CodeChangeDataset import CodeChangeDataset

logger = logging.getLogger(__name__)


class JITDPDataset(CodeChangeDataset):
    def __init__(self, tokenizer, args, file_path=None, mode='train'):
        super().__init__(tokenizer, args, file_path, mode)

    def __getitem__(self, item):
        return {"text": self.examples[item].text,
                "labels": self.examples[item].label,
                "idx": item,
                "manual_features": self.examples[item].manual_features,
                "commit_id": self.examples[item].commit_id,
                }

