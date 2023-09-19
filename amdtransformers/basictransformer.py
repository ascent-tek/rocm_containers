from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time

model = "tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
start_time = time.time()
sequences = pipeline(
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
    max_length=400,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
stop_time = time.time()
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

elapsed_time= stop_time - start_time
print(f"Generation Time: {elapsed_time:.4f} seconds")
