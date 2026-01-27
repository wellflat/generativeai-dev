#!/usr/bin/env python

import os
import sys
from pprint import pprint

import requests
from dotenv_flow import dotenv_flow

# Load environment variables
dotenv_flow("dev")

api_key = os.environ.get("CEREBRAS_API_KEY")
url = "https://api.cerebras.ai/v1/models"

if not api_key:
    print("Error: Environment variable CEREBRAS_API_KEY is not set.")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

try:
    print(f"Fetching models from {url}...")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()
    print("\n--- Available Models ---")
    pprint(data)

except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")
