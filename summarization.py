# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from Vector_DB import vector_db
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("Falconsai/text_summarization").to(device)

class Summarizer():
    def __init__(self):
        self.tokenizer = tokenizer
        self.model = model

    def summarize(self, chunks):
        
        list_of_summary = []
        for chunk in chunks:
            print(chunk)
            input_ids = tokenizer.encode(chunk, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(input_ids, num_beams=4, early_stopping=True, max_length=256)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            list_of_summary.append(summary)
        
        summary = "\n\n".join(list_of_summary)

        return summary
    

    

