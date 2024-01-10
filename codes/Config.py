import sys
from os import path


basedir = path.abspath(path.dirname(__file__))
codes_dir = basedir
project_dir = path.join(basedir, "../")
sys.path.append(project_dir)  # Change to root directory of your project
sys.path.append(codes_dir)  # Change to root directory of your project

from transformers import BertTokenizer, BertForMaskedLM, RobertaForMaskedLM, RobertaTokenizer, BertTokenizerFast, \
    RobertaForSequenceClassification
from codes.Models.ModelConcat import ModelConcat
from codes.Models.ModelSemantic import ModelSemantic


class Config:
    OUTPUT_DIR = "output"

    MODEL_CLASSES = {"BertMLM": (BertForMaskedLM, BertTokenizer),
                     "RobertaMLM": (RobertaForMaskedLM, RobertaTokenizer),
                     "RobertaSequenceClassification": (RobertaForSequenceClassification, RobertaTokenizer),
                     "JITDPSemantic": (ModelSemantic, RobertaTokenizer),
                     "JITDPConcat": (ModelConcat, RobertaTokenizer),
                     }

    def __init__(self):
        basedir = path.abspath(path.dirname(__file__))
        self.codes_dir = basedir
        self.project_dir = path.join(basedir, "../")

    def append_path(self):
        sys.path.append(self.project_dir)  # Change to root directory of your project
        sys.path.append(self.codes_dir)  # Change to root directory of your project


cfg = Config()
cfg.append_path()
