import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import warnings

# Load environment variables from .env
load_dotenv()


# Define conversation template
chat_template = """
**You:** {user_input}
**Gemma-7b:** {model_response}
"""

# Load pre-trained model and tokenizer
model_id= "google/gemma-7b-it"
lora_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
quantized_model = AutoModelForCausalLM.from_pretrained(model_id)

warnings.filterwarnings('ignore')


def chatbot():
    """
    Interactive chat function with Gemma-2b chatbot.
    """
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # Encode user input with tokenizer
        input_ids = tokenizer.encode(user_input, return_tensors="pt")

        # Generate model response
        output = quantized_model.generate(input_ids, max_length=1024, do_sample=True, top_k=10, top_p=0.9)
        model_response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Print conversation
        print("Gemma-7b ---> ", model_response.split("\n")[-1])


if __name__ == "__main__":
    print("Welcome to the Gemma-2b Chatbot!")
    chatbot()
    print("Goodbye!")
