# from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from summarization import Summarizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(device)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

summarization_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an advanced language model trained to generate high-quality summaries. Your task is to condense the provided text into a clear and concise summary that captures the essential information and main ideas. Follow these instructions carefully:

1. Understand the Text: Read the entire text thoroughly to grasp the main points and overall context.
2. Extract Key Points: Identify the most important details, including significant events, key facts, primary arguments, and essential information.
3. Summarize Effectively:
   - Be Concise: Reduce the text to its core components, removing unnecessary details, examples, and repetitions.
   - Maintain Accuracy: Ensure the summary accurately reflects the original text without misrepresenting any information.
4. Use Clear Language: Write in a way that is easy to understand, maintaining logical flow and coherence.
5. Paraphrase: Use your own words to rephrase the content, avoiding direct copying of sentences from the original text unless quoting is necessary.
6. Objective Tone: Keep the summary factual and objective, avoiding any personal opinions or interpretations.

### Input:
{}

### Response:
{}"""

class inf_model(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("MahmoudMohamed/Phi3_DPO_MeetingQA_4bit")
        self.model = AutoModelForCausalLM.from_pretrained("MahmoudMohamed/Phi3_DPO_MeetingQA_4bit")

    def predict(self, query, transcript):
        # Tokenize the input text with truncation and add padding if necessary
        inputs = self.tokenizer(
            [
                alpaca_prompt.format(
                    query, # instruction
                    transcript, # input
                    "", # output - leave this blank for generation!
                )
            ], return_tensors="pt").to(device)  # Move inputs to the device

        # Generate text using the model with max_new_tokens to avoid length issues
        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024  
        )

        # Decode the generated text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def llm_summarize(self, transcript):

        

        inputs = self.tokenizer(
            [
                summarization_prompt.format(
                    transcript,
                    "", # output - leave this blank for generation!
                )
            ], return_tensors="pt").to(device)  # Move inputs to the device
        print("llm_summarize")
        # Generate text using the model with max_new_tokens to avoid length issues
        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024  
        )

        # Decode the generated text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)        
        
        
    