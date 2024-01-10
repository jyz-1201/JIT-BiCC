import random

import math
import logging

from codes.Dataset.CodeChangeDataset import CodeChangeDataset

logger = logging.getLogger(__name__)


class RNMIDataset(CodeChangeDataset):
    def __init__(self, tokenizer, args, file_path=None, mode='train'):
        super().__init__(tokenizer, args, file_path, mode)

        def get_masked_msg(msg: str, code: str):
            msg_tokens = msg.lower().split()
            code_tokens = code.lower().split()
            idx_to_mask = []
            mask_token_num = math.floor(len(msg_tokens) * args.RNMI_masking_ratio)
            for idx, msg_token in enumerate(msg_tokens):
                if msg_token in code_tokens:
                    idx_to_mask.append(idx)
            random.shuffle(idx_to_mask)

            if len(idx_to_mask) > mask_token_num:
                idx_to_mask = idx_to_mask[:mask_token_num]
            elif len(idx_to_mask) < mask_token_num:
                all_idx = [i for i in range(len(msg_tokens))]
                idx_not_masked = [x for x in all_idx if x not in idx_to_mask]
                random.shuffle(idx_not_masked)
                for idx_to_append in idx_not_masked[: mask_token_num - len(idx_to_mask)]:
                    idx_to_mask.append(idx_to_append)

            msg_tokens = [self.tokenizer.mask_token if idx in idx_to_mask else token
                          for idx, token in enumerate(msg_tokens)]
            return " ".join(msg_tokens)

        self.RNMI_msg = []
        self.RNMI_masked_msg = []
        self.RNMI_code = []
        for example in self.examples:
            text = example.text
            msg_boundary = text.find("[ADD]")
            if msg_boundary == -1:
                msg_boundary = text.find("[DEL]")
            msg = text[: msg_boundary]
            code = text[msg_boundary:]
            self.RNMI_msg.append(msg)
            self.RNMI_code.append(code)
            self.RNMI_masked_msg.append(get_masked_msg(msg, code))

    def __getitem__(self, item):
        do_replace = random.randint(0, 1)
        if do_replace:
            new_msg_idx = random.randint(0, len(self.RNMI_msg) - 1)
            while new_msg_idx == item:
                new_msg_idx = random.randint(0, len(self.RNMI_msg) - 1)
        else:
            new_msg_idx = item
        text = self.RNMI_masked_msg[new_msg_idx] + self.RNMI_code[item]
        return {"text": text,
                "labels": do_replace,
                "idx": item + (len(self.RNMI_msg) if do_replace else 0),
                "commit_id": self.examples[item].commit_id,
                }



