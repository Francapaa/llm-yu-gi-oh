from abc import ABC, abstractmethod
import enum
from pathlib import Path
import json
from typing import Optional
from torch import Tensor


CARD_START = "<card>"
CARD_END = "</card>"



class TokenizerType(enum): 
   CHAR = "char"
   BPE = "bpe" # byte pair encode


class Tokenizer(ABC): 
    @abstractmethod
    def encode(self, s: str) -> Tensor:
        pass
    
    @abstractmethod
    def decode(self, ix: Tensor)-> str:
        pass

    @abstractmethod
    def vocab_size(self) -> int:
        pass
    
    @abstractmethod
    def get_types(self) -> TokenizerType:
        pass



class CharacterTokenizer(Tokenizer):
    def __init__(self, mapping_path: Optional[Path] = None):
        self.mapping_path = mapping_path
        self.c_to_i = {}
        self.i_to_c = {}
        if mapping_path:
            with open(mapping_path, "r") as f:
                self.c_to_i = json.load(f)
                for c, i in self.c_to_i.items():
                    self.i_to_c[i] = c
    
    def building_mapping(self, input_dataset_path: Path, mapping_save_path: Path):
        with open(input_dataset_path, "r") as f:
            dataset_str = f.read()
        
        c_to_i = {c: i for i, c in enumerate(sorted(set(dataset_str)))}
        
        with open(mapping_save_path, "w") as f:
            json.dump(c_to_i, f, ensure_ascii=False)
            print(f"Total tokenizer keys {len(c_to_i)}")

    def vocab_size(self) -> int:
        if not self.c_to_i:
            raise Exception("Forgot to load tokenizer mapping")
        return len(self.c_to_i)
        