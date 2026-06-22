from abc import ABC, abstractmethod
import enum
from pathlib import Path
import json
from typing import Optional
from torch import Tensor, tensor
from tokenizers import  Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, trainers, processors, decoders

CARD_START = "<card>"
CARD_END = "</card>"



class TokenizerType(enum.Enum): 
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



class BPETokenizer(Tokenizer):
    def __init__(self, mapping_path: Optional[Path] = None):
        self.mapping_path = mapping_path
        if self.mapping_path:
            self.tokenizer = HFTokenizer.from_file(str(self.mapping_path))
        else:
            self.tokenizer = HFTokenizer(models.BPE())


    def building_mapping( self, input_dataset_path: Path, mapping_save_path: Path, vocab_size: int):
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=[CARD_START, CARD_END], 
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self.tokenizer.train([str(input_dataset_path)], trainer=trainer)
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.save(str(mapping_save_path))
        print(f"Saved tokenizer mapping to {mapping_save_path}")


    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, s: str) -> int:
        return tensor(self.tokenizer.encode(s).ids)

    def decode(self, ix: Tensor) -> str:
        return self.tokenizer.decode(ix.tolist(), skip_special_tokens=False)

    def get_type(self) -> TokenizerType:
        return TokenizerType.BPE 

    