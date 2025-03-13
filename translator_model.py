from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_core.prompts import PromptTemplate
from accelerate import infer_auto_device_map, dispatch_model
import warnings
warnings.filterwarnings('ignore')

model_name = "abhinand/tamil-llama-7b-instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype = torch.float16,
                                             device_map='auto',
                                             offload_folder="abhinand/offload" )
# offload_dir = "offload"
# device_map = infer_auto_device_map(model)
# model = dispatch_model(model,device_map, offload_dir=offload_dir)


# template = PromptTemplate.from_template("""You are a highly knowledgeable technical assistant specializing in AI. 
# Your task is to assist users by answering their questions, writing code, and explaining both theory and code in the AI domain.

# Question: {question}

# Provide a well-structured response, including:
# 1. A clear and concise explanation.
# 2. If applicable, relevant Python code with comments.
# 3. A breakdown of the code explaining its functionality.

# Answer:
# """)

def format_prompt(system_prompt="You are a language translator.",
                  instruction = "Translate from English to Tamil", 
                  input_text=None):
    if input_text:
        prompt = f"""{system_prompt}

### Instruction:
{instruction}

### Input:
{input_text}

### Response:"""
    else:
        prompt = f"""{system_prompt}

### Instruction:
{instruction}

### Response:"""

    return prompt


def generate_response(question, system_prompt="You are a language translator.",
                      instruction="Translate from English to Tamil",
                      max_length=500):
    # Properly format the prompt
    prompt = format_prompt(system_prompt=system_prompt, instruction=instruction, input_text=question)
    

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    

    # Extract only the generated response
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


