from transformers import pipeline

# https://huggingface.co/models


# The easiest way to use HF is via Pipelines: https://huggingface.co/docs/transformers/main_classes/pipelines
generator = pipeline(task="text-generation") # https://huggingface.co/tasks
output = generator("Who won the world series in 2020?")


# Pipelines can be used with any model from HF. In the following is an example
# for text generation with Bloomz: https://huggingface.co/bigscience/bloomz

# Load Bloomz from HF
generator_bloomz = pipeline(model="bigscience/bloomz-1b7")
output_bloomz = generator_bloomz("Who won the world series in 2020?")


# -------------------------------Phi-2 OpenHermes-2.5--------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cpu") # "cuda" or "cpu"

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)


# -------------------------------Phi-2 Fine Tuned on OpenHermes-2.5-----------------------------


modelpath = "g-ronimo/phi-2-OpenHermes-2.5"

model_2 = AutoModelForCausalLM.from_pretrained(
    modelpath,
    torch_dtype=torch.bfloat16,
    device_map="cpu" # "cuda" or "auto" or "cpu"
)
tokenizer_2 = AutoTokenizer.from_pretrained(modelpath)

messages = [
    {"role": "system", "content": "answer like a pirate"},
    {"role": "user", "content": "what does it mean to be successful?"},
]

input_tokens = tokenizer_2.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
)

output_tokens = model_2.generate(input_tokens, max_new_tokens=500)
output = tokenizer_2.decode(output_tokens[0])

print(output)

