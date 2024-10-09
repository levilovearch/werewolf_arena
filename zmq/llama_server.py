import zmq
from transformers import pipeline
import json

def main():
    # Load a pre-trained transformers model (e.g., text generation using GPT-2)
    model = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", device_map="auto")

    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a REP (reply) socket
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:7555")  # Bind to port 5555 on all network interfaces

    print("Server is running and waiting for messages...")

    while True:
        try:
            # Wait for the next request from the client
            message = socket.recv_string()
            print(f"Received request: {message}")
            messages = []
            messages.append(json.loads(message))
            # Use the model to generate a response
            conversation = model(messages, max_new_tokens=1024, num_return_sequences=1)
            
            response = conversation[0]["generated_text"][-1]["content"]
            # Send the generated text back to the client
            socket.send_string(response)
            print(f"Sent reply: {response}")

        except KeyboardInterrupt:
            print("\nServer is shutting down.")
            break

    # Clean up
    socket.close()
    context.term()

if __name__ == "__main__":
    main()
