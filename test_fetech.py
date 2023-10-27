import requests

# Replace with the actual URL of your Flask API
api_url = "http://127.0.0.1:5000/detect"  # Change the port if necessary

# Replace with the path to the image file you want to send
image_file = "Purchase Order (No. DD111822-6) from Devi Designs LLC_page0.jpg"

# Create a dictionary with the image file
# files = {"pdf": ("test.pdf", open(pdf_file, "rb"))}
files = {"image_file": ("image.jpg", open(image_file, "rb"))}
# Inference arguments (optional)
# data = {"size": 640, "confidence": 0.85, "iou": 0.50}

# Send a POST request to the API
response = requests.post(api_url, files=files)
# response = requests.post(api_url, files=files, data=data)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # Assuming the API returns JSON data
    print("API Response:", data)
else:
    print("API request failed with status code:", response.status_code)
