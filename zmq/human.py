import zmq

def main():
    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a REP (reply) socket
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5556")  # Bind to port 5556 on all network interfaces

    print("Server is running and waiting for messages...")

    while True:
        try:
            # Wait for the next request from the client
            message = socket.recv_string()
            print(f"Received request: {message}")

            # Ask the human to input a response
            response = input("Please type your response: ")

            # Send the response back to the client
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
