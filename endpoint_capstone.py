import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://e8a9175a-fc05-4608-aef8-f19fbc252ffb.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = '6Pw22x1AEL0UqLbyso9EjGK0HiybwZhs'

# Convert to JSON string
with open("input_data.json", "w") as _f:
	input_data_json = json.load(f)
    

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
#headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data_json, headers=headers)
print(resp.json())