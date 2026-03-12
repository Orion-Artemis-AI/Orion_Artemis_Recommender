from llama_cpp import Llama


model_path = "D:\AITF_26_Model\kalm-embedding-multilingual-mini-instruct-v2.5-q8_0.gguf"

llm = Llama(model_path=model_path,               
            flash_attn=True,
            embedding=True,
            n_gpu_layers=0,
            n_batch=1024,
            n_ctx=1024,
            n_threads=6,
            n_threads_batch=6,
            )

# Prompt creation
system_message = "Kamu adalah asisten"
user_message = "siapa kamu?"

prompt = f"""
{user_message}"""

# Run the model
output = llm(
    prompt, # Prompt 
    max_tokens = 64,
    temperature = 1.0,
    top_p = 0.95,
    top_k = 40,
    repeat_penalty = 1.1,
    echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion

print(output["choices"][0]["text"])