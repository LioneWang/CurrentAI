# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Set a seed for reproducibility
set_seed(2024)

# Define the prompt for the model
prompt = "Do u know the highest tower in France?"

# Define the model checkpoint simply replace with Phi-3 Model Required
model_checkpoint = "./microsoft/Phi-3.5-mini-instruct"

# Load the tokenizer from the model checkpoint
# trust_remote_code=True allows the execution of code from the model files
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)

# Load the model from the model checkpoint
# trust_remote_code=True allows the execution of code from the model files
# torch_dtype="auto" automatically determines the appropriate torch.dtype
# device_map="cuda" specifies that the model should be loaded to the GPU
model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                             trust_remote_code=True,
                                             torch_dtype="auto",
                                             device_map="cuda")

# Tokenize the prompt and move the tensors to the GPU
inputs = tokenizer(prompt,
                   return_tensors="pt").to("cuda")

# Generate a response from the model
# do_sample=True means the model will generate text by sampling from the distribution of possible outputs
# max_new_tokens=120 limits the length of the generated text to 120 tokens
outputs = model.generate(**inputs,
                         do_sample=True, max_new_tokens=120)

# Decode the generated tokens and remove any special tokens
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)