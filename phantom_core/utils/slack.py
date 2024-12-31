import requests
import os

def send_slack_message(message):
    # Get the webhook URL from environment variables
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    
    if not webhook_url:
        raise ValueError("SLACK_WEBHOOK_URL environment variable is not set")

    # Prepare the payload
    payload = {'text': message}

    # Send the POST request to Slack
    response = requests.post(webhook_url, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        print("Message sent successfully to #pilot-model")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")
        print(f"Response: {response.text}")