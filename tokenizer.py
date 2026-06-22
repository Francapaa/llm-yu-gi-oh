from abc import abstractmethod
import enum

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
    