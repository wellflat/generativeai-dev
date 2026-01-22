#!/usr/bin/env python

import os

import requests
from dotenv_flow import dotenv_flow
from pprint import pprint

dotenv_flow("dev")

api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers, timeout=30)
pprint(response.json())
