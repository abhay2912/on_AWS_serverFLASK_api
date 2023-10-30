from flask import Flask, request, jsonify
import torch
import json
from PIL import Image
import numpy as np
# from torchvision.transforms import functional as F
import pytesseract
import torch

pytesseract.pytesseract.tesseract_cmd='/usr/bin/tesseract'  # Replace with your Tesseract installation path

app = Flask(__name__)

# Define the paths to your YOLOv5 model weights
YOLO_TABLE_MODEL_PATH = 'models/table/best.pt'
YOLO_PAGE_MODEL_PATH = 'models/page/best.pt'


model_table = torch.hub.load('ultralytics/yolov5', 'custom', path='models/table/best.pt')  # local model
model_pagee = torch.hub.load('ultralytics/yolov5', 'custom', path='models/page/best.pt')
classes = {
    0: "issuer",
    1: "issuer_address",
    2: "supplier",
    3: "ship_to",
    4: "po_number",
    5: "issue_date",
    6: "total_quantity",
    7: "total_amount",
    8: "table",
    9: "vendor",
    10: "bill_to",
    11: "head",
    12: "issuer_phone_number",
    13: "factory",
    14: "customer",
    15: "shipping_date"
}


print('loaded model')
# model_table = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, path=YOLO_TABLE_MODEL_PATH)

def prediction_table(image, model, conf_threshold=0.85):
    print(type(image))
    pil_image = Image.fromarray(image)

    # tensor_a = torch.from_numpy(image)
    # print(type(tensor))
    # image = Image.fromarray(PIL_image.astype('uint8'), 'RGB')
    # print(type(pil_image))
    # image = Image.open(pil_image)
    results = model(image)

    # print(results.xyxy[0])

    tensor = results.xyxy[0]

    output = []
    centerpoint_list = []
    for box in tensor:
        x1, y1, x2, y2, confidence, class_id = box[:6]  # Extract relevant values
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = round(confidence.item(), 2)

        # Append the information to the output list
        output.append([x1, y1, x2, y2, class_id, confidence])

        # Calculate the center point and add it to the centerpoint_list
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # Extract the region of interest (roi) from the image using slicing
        image_np = np.array(image)
        roi = image_np[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi)

        centerpoint_list.append([center_x, center_y, text])

    # Print the center points
    for item in centerpoint_list:
        print("Center X:", item[0], "Center Y:", item[1], "Class:", item[2])


    y_sorted_list=sorted(centerpoint_list, key=lambda x:x[1])
    matrix=[]
    sorted_matrix=[]
    element=y_sorted_list[0]
    for i in y_sorted_list:
        if i[1]<=element[1]+10:
            end=y_sorted_list.index(i)
            row=y_sorted_list[y_sorted_list.index(element):end+1]
            # print(row)

        else:
            matrix.append(row)
            element=i

    output_text = ""
    for i in matrix:
        i=sorted(i,key=lambda x:x[0])
        #print(i)
        sorted_matrix.append(i)

    # print("sorted list")
    # print(sorted_matrix)
    # # -------------------------------joson fromat-----------------------------
    return convert_data_to_json(sorted_matrix)


def convert_data_to_json(data):
    keys = [cell[2].strip() for cell in data[0]]
    product_data = {}

    for i, row in enumerate(data[2:]):
        product = {}
        for j, cell in enumerate(row):
            key = keys[j]
            value = cell[2].strip()
            if value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            product[key] = value
        product_data[f"productID_{i+1}"] = product

    return product_data

def jsonify_page(data):
    # Initialize an empty dictionary to store the key-value pairs
    json_data = {}

    for item in data:
        # Extract the key and value from each sublist
        key = item[2]
        value = item[3]

      # Check if the key is a list and take the first item as the dictionary key
        if isinstance(key, list):
            key = key[0]

        # Check if the value is a string before performing operations on it
        if isinstance(value, str):
            # Replace newline characters with spaces and strip leading/trailing whitespace
            value = value.replace('\n', ' ').strip()
        # print(key, value)
        # Add the key-value pair to the JSON dat
        json_data[key] = value

    return json_data



def predicition_page(image, model, conf_threshold=0.85):

    having_table  = False
    print(type(image))
    image = Image.open(image)
    print(type(image))
    results = model(image)
    tensor = results.xyxy[0]
    output = []
    centerpoint_list = []
    image_np = np.array(image)
    print(tensor)
    for box in tensor:
        x1 , y1, x2, y2, confidence, classID = box[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = round(confidence.item(), 2)

        table_roi = image_np[y1:y2, x1:x2]
        # classID = int(classID)
        print("CLASS ID")
        print(classID.item())
        if(classID==8):
            table_json = prediction_table(table_roi,model_table)
            # print(table_json)
            having_table  = True
            continue
        center_x = int((x1+x2)/2)
        center_y = int((y1+y2)/2)
        # print("NOT a table")
        # image_np = np.array(image)
        each_class_roi = image_np[y1:y2, x1:x2]
        text = pytesseract.image_to_string(each_class_roi)
        centerpoint_list.append([center_x, center_y,classes[classID.item()], text])
    page_json = jsonify_page(centerpoint_list)

    if(having_table):
        # Create the combined JSON
        combine_json = {
            **page_json,  # Include all key-value pairs from json_1
            'table': {
                'Total_product': len(table_json),  # Number of products in json_2
                'products': table_json  # Include json_2 as 'products'
            }
        }
        return combine_json
    else: return (page_json)
    # return centerpoint_list

@app.route("/detect", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", passes it
        through YOLOv5 object detection network and returns the data on performing
        tesseract of the elements of the table.
        :return: a JSON array of element bounding box bounding
    """
    if "image_file" not in request.files:
        return "No 'image_file' in request", 400

    image_file = request.files["image_file"]
    if image_file.filename == "":
        return "No selected file", 400

    # Call your object detection function with the image stream
    # json_result = prediction_tabel(image_file.stream, model_table)
    json_result = predicition_page(image_file.stream, model_pagee)

    print(json.dumps(json_result, indent=2))

    return json.dumps(json_result, indent=2)

if __name__ == '__main__':
    app.run(debug=True)