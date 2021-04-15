import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://e8a9175a-fc05-4608-aef8-f19fbc252ffb.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = '6Pw22x1AEL0UqLbyso9EjGK0HiybwZhs'

# Two sets of data to score, so we get two results back
data = {"data":
        [{
                "age":55
		"anaemia":0
		"creatinine_phosphokinase":981
		"diabetes":0
		"ejection_fraction":50
		"high_blood_pressure":1
		"platelets":265000
		"serum_creatinine":135
		"serum_sodium":132
		"sex":0
		"smoking":1
		"time":80}
        ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())