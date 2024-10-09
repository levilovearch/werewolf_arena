# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from openai import OpenAI
import os
import subprocess

from typing import Any
import google
import vertexai
from vertexai.preview import generative_models
from anthropic import AnthropicVertex
from huggingface_hub import InferenceClient
import zmq
import json
import requests


def generate(model, **kwargs):
    if "gpt" in model:
        return generate_openai(model, **kwargs)
    elif "claude" in model:
        return generate_authropic(model, **kwargs)
    elif "zmq" in model or "llama" in model:
        return generate_hf(model, **kwargs)
    elif "human" in model:
        return generate_human(model, **kwargs)
    else:
        return generate_vertexai(model, **kwargs)


# openai
def generate_openai(model: str, prompt: str, json_mode: bool = True, **kwargs):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response_format = {"type": "text"}
    if json_mode:
        response_format = {"type": "json_object"}
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        model=model,
    )

    txt = response.choices[0].message.content
    return txt


# anthropic
def generate_authropic(model: str, prompt: str, **kwargs):
    # For local development, run `gcloud auth application-default login` first to
    # create the application default credentials, which will be picked up
    # automatically here.
    _, project_id = google.auth.default()
    client = AnthropicVertex(region="us-east5", project_id=project_id)

    response = client.messages.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=1024
    )

    return response.content[0].text


# vertexai
def generate_vertexai(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    json_mode: bool = True,
    json_schema: dict[str, Any] | None = None,
    **kwargs,
) -> str:
    """Generates text content using Vertex AI."""

    # For local development, run `gcloud auth application-default login` first to
    # create the application default credentials, which will be picked up
    # automatically here.
    credentials, project_id = google.auth.default()

    vertexai.init(
        project=project_id,
        location="us-central1",
        credentials=credentials,
    )
    model_endpoint = generative_models.GenerativeModel(model)

    # 1.5 flash doesn't support constrained decoding as of 6/5/2024, so we
    # disable json_schema for it. Otherwise, the library will throw an unsupported
    # error.
    if "flash" in model:
        json_schema = None

    response_mimetype = None
    if json_mode or json_schema is not None:
        response_mimetype = "application/json"
    config = generative_models.GenerationConfig(
        temperature=temperature,
        response_mime_type=response_mimetype,
        response_schema=json_schema,
    )

    # Safety config.
    safety_config = [
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
        ),
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
        ),
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
        ),
        generative_models.SafetySetting(
            category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]
    response = model_endpoint.generate_content(
        prompt,
        generation_config=config,
        stream=False,
        safety_settings=safety_config,
    )
    assert isinstance(response, generative_models.GenerationResponse)

    return response.text

def generate_hf(model: str, prompt: str, **kwargs):
    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a REQ (request) socket
    socket = context.socket(zmq.REQ)
    if "llama" in model:
        socket.connect("tcp://localhost:7555")
    else:
        socket.connect("tcp://localhost:5555")  # Connect to the server

    message = {"role": "user", "content": prompt}
    message_str = json.dumps(message)
    # print(f"Sending request: {message_str}")
    socket.send_string(message_str)

    response = socket.recv_string()
    socket.close()
    context.term()
    return response

def generate_human(model: str, prompt: str, **kwargs):
    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a REQ (request) socket
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")  # Connect to the server

    # print(f"Sending request: {message_str}")
    socket.send_string(prompt)

    response = socket.recv_string()
    socket.close()
    context.term()
    return response

def generate_remote_hf(model: str, prompt: str, **kwargs):
    # Create a ZeroMQ context
    context = zmq.Context()

    # Create a REQ (request) socket
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://moore:5555")  # Connect to the server

    message = {"role": "user", "content": prompt}
    message_str = json.dumps(message)
    # print(f"Sending request: {message_str}")
    socket.send_string(message_str)

    response = socket.recv_string()
    socket.close()
    context.term()
    return response

def generate_hf_inference(model: str, prompt: str, **kwargs):
    client = InferenceClient(token="")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
    )
    txt = response.choices[0].message.content
    print(txt)
    if txt.endswith("\""):
        txt = txt[:-1]
    elif txt.endswith("```"):
        txt = txt[:-3]
    elif txt.endswith("\"```"):
        txt = txt[:-4]
    return txt

def generate_vertex_post(model: str, prompt: str, **kwargs):
    # Replace these with your actual values
    ENDPOINT = 'us-central1-aiplatform.googleapis.com'
    PROJECT_ID = 'eminent-citadel-271315'
    REGION = 'us-central1'

    # Get the access token using gcloud
    # Get the access token using gcloud
    access_token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token"]
    ).strip().decode('utf-8')

    # API endpoint
    url = f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

    # Headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Payload (data to send)
    payload = {
        "model": "meta/llama3-405b-instruct-maas",
        "stream": True,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    # Step 1: Split the response into lines (each line contains a JSON object)
    lines = response.text.splitlines()
    # Step 2: Loop through the lines and extract the content
    responses = ""
    for line in lines:
        if line.startswith('data:'):
            json_str = line[6:].strip()  # Remove the 'data: ' prefix
            try:
                data = json.loads(json_str)  # Convert the string to a JSON object
                # Extract the 'content' field
                if 'choices' in data:
                    for choice in data['choices']:
                        content = choice.get('delta', {}).get('content', "")
                        if content:
                            responses+=content
            except json.JSONDecodeError:
                continue
    first_brace = responses.find('{')  
    last_brace = responses.rfind('}')
    responses = responses[first_brace:last_brace+1]
    print(f"Api return:{responses}")
    return responses