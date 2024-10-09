import zmq
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    # Load the quantized model and tokenizer
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load the model with quantization and device mapping
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        quantization_config=quantization_config
    )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a REP (reply) socket
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:6555")  # Bind to port 5555 on all network interfaces

    print("Server is running and waiting for messages...")

    while True:
        try:
            # Wait for the next request from the client
            message = socket.recv_string()
            print(f"Received request: {message}")
            messages = []
            # Parse the input message (assuming JSON structure)
            messages.append(json.loads(message))
            
            # Tokenize the input text with attention mask
            # inputs = tokenizer(messages, return_tensors="pt").to("cuda")
            tokenized_chat = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt",
                padding=True,  # Ensure proper padding
                truncation=True  # Ensure truncation if the sequence is too long
            ).to(quantized_model.device)
            input_data_offset = tokenized_chat.shape[1]
            # Generate attention mask for padding tokens
            # attention_mask = tokenized_chat['attention_mask']
            with torch.no_grad():
                output = quantized_model.generate(
                    tokenized_chat,
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Generate the output using the quantized model with attention mask
            # output = quantized_model.generate(
            #     tokenized_chat['input_ids'], 
            #     attention_mask=attention_mask,  # Pass the attention mask
            #     max_new_tokens=1024, 
            #     pad_token_id=tokenizer.eos_token_id,
            #     eos_token_id=tokenizer.eos_token_id
            # )


            # Decode the generated output
            response_text = tokenizer.decode(output[0][input_data_offset:], skip_special_tokens=True)
        
            # Send the generated text back to the client
            if response_text.endswith("\""):
                response_text = response_text[:-1]
            socket.send_string(response_text)
            print(f"Sent reply: {response_text}")

        except KeyboardInterrupt:
            print("\nServer is shutting down.")
            break


    # Clean up
    socket.close()
    context.term()

if __name__ == "__main__":
    main()
