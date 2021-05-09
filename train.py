from transformers import * 
import os 

def main(): 
    tokenizer = GPT2Tokenizer.from_pretrained('ckpt', do_lower_case=True)
    model = GPT2Model.from_pretrained('ckpt')


if __name__ == '__main__': 
    main()
