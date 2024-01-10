import random

import math
import logging

from codes.Dataset.CodeChangeDataset import CodeChangeDataset
from codes.utils.OtherUtils import return_true_with_probability

logger = logging.getLogger(__name__)


class RDNMIDataset(CodeChangeDataset):
    def __init__(self, tokenizer, args, file_path=None, mode='train'):
        super().__init__(tokenizer, args, file_path, mode)

        def get_idx_to_mask(msg: str, code: str):
            msg_tokens = self.get_msg_tokens(msg)
            code_tokens = self.get_code_tokens(code)
            idx_to_mask = []
            for idx, msg_token in enumerate(msg_tokens):
                if any(c.isalpha() for c in msg_token):
                    if msg_token in code_tokens:
                        idx_to_mask.append(idx)
            return idx_to_mask

        self.RNMI_msg = []
        self.RNMI_idx_to_mask = []
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
            self.RNMI_idx_to_mask.append(get_idx_to_mask(msg, code))

    @staticmethod
    def get_msg_tokens(msg: str):
        msg = msg.replace("<s>", "")
        msg_tokens = msg.lower().split()
        return msg_tokens

    @staticmethod
    def get_code_tokens(code: str):
        code = code.replace("[ADD]", " ")
        code = code.replace("[DEL]", " ")
        code = code.replace("</s>", "")
        code_tokens = code.lower().split()
        return code_tokens

    def __getitem__(self, item):
        do_replace = random.randint(0, 1)
        if do_replace:
            new_msg_idx = random.randint(0, len(self.RNMI_msg) - 1)
            while new_msg_idx == item:
                new_msg_idx = random.randint(0, len(self.RNMI_msg) - 1)
        else:
            new_msg_idx = item
        text = self.get_masked_msg(new_msg_idx) + self.RNMI_code[item]
        return {"text": text,
                "labels": do_replace,
                "idx": item + (len(self.RNMI_msg) if do_replace else 0),
                "commit_id": self.examples[item].commit_id,
                }

    def get_masked_msg(self, msg_idx: int):
        idx_to_mask = self.RNMI_idx_to_mask[msg_idx]
        msg_tokens = self.get_msg_tokens(self.RNMI_msg[msg_idx])
        random.shuffle(idx_to_mask)
        mask_token_num = math.floor(len(msg_tokens) * self.args.RNMI_noise_ratio)

        if len(idx_to_mask) > mask_token_num:
            idx_to_mask = idx_to_mask[:mask_token_num]
        elif len(idx_to_mask) < mask_token_num:
            all_idx = [i for i in range(len(msg_tokens))]
            idx_not_masked = [x for x in all_idx if x not in idx_to_mask]
            random.shuffle(idx_not_masked)
            for idx_to_append in idx_not_masked[: mask_token_num - len(idx_to_mask)]:
                idx_to_mask.append(idx_to_append)

        noisy_msg_tokens = []

        for idx, token in enumerate(msg_tokens):
            if idx in idx_to_mask:
                if return_true_with_probability(self.args.RNMI_masking_ratio):
                    noisy_msg_tokens.append(self.tokenizer.mask_token)
            else:
                noisy_msg_tokens.append(token)

        res = "<s>" + (" ".join(noisy_msg_tokens))
        return res


