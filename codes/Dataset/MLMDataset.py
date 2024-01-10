from codes.Dataset.CodeChangeDataset import CodeChangeDataset


class MLMDataset(CodeChangeDataset):
    def __init__(self, tokenizer, args, file_path, mode):
        super().__init__(tokenizer, args, file_path, mode)

    def __getitem__(self, item):
        return {"text": self.examples[item].text}


