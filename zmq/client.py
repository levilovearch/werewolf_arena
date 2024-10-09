import zmq
import json

def main():
    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a REQ (request) socket
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")  # Connect to the server

    message = {"role": "user", "content": "how are you?"}
    message_str = json.dumps(message)
    # Prepare the message to send
    # prompt = {"role": "user", "content": "how are you?"}
    print(f"Sending request: {message_str}")
    socket.send_string(message_str)

    try:
        # Wait for the reply from the server
        response = socket.recv_string()
        print(f"Received reply: {response}")
    except KeyboardInterrupt:
        print("\nClient is shutting down.")

    # Clean up
    socket.close()
    context.term()

if __name__ == "__main__":
    main()
