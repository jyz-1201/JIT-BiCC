import random

from codes.Dataset.CodeChangeDataset import CodeChangeDataset


class DDTPDataset(CodeChangeDataset):
    def __init__(self, tokenizer, args, file_path, mode):
        super().__init__(tokenizer, args, file_path, mode)
        self.DDTP_msgs = []
        self.DDTP_code_lines = []

        for example in self.examples:
            text = example.text
            msg_boundary = text.find("[ADD]")
            if msg_boundary == -1:
                msg_boundary = text.find("[DEL]")
            msg = text[: msg_boundary]
            code = text[msg_boundary:]
            code_lines = []
            start_pos = last_pos = 0
            while True:
                res = code.find("[ADD]", start_pos + 5)
                if res == -1:
                    res = code.find("[DEL]", start_pos + 5)
                if res != -1:
                    code_lines.append(code[start_pos: res])
                    start_pos = res
                    last_pos = start_pos
                else:
                    break
            self.DDTP_msgs.append(msg)
            code_lines.append(code[last_pos: -4])
            self.DDTP_code_lines.append(code_lines)

    def __getitem__(self, item):
        code_lines = self.DDTP_code_lines[item]
        random.shuffle(code_lines)
        shuffled_text = self.DDTP_msgs[item] + "".join(code_lines) + "</s>"

        return {"text": shuffled_text}


