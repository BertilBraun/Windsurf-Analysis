# make a request to the backend
# https://bertil-braun-private--windsurf-analysis-fastapi-app.modal.run/api/v1/admin/users
# with the following payload:
# {
#     "secret": "secret",
#     "email": "test@test.com",
#     "password": "test"
# }

import os
import requests
import dotenv

dotenv.load_dotenv('server/.env')

url = os.getenv('BACKEND_PUBLIC_BASE_URL') + '/v1/admin/users'
secret = os.getenv('USER_CREATE_SECRET')
email = 'test@web.com'
password = 'password'

payload = {'secret': secret, 'email': email, 'password': password}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.text)
