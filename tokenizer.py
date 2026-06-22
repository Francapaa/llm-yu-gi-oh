import enum

CARD_START = "<card>"
CARD_END = "</card>"



class Tokenizer(enum): 
   CHAR = "char"
   BPE = "bpe" # byte pair encode