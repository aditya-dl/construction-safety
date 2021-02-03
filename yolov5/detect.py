# import dependencies
import numpy as np 
import cv2
import torch

def load_model(weights='yolov5/weights/last.pt'):
    return torch.hub.load('ultralytics/yolov5', 'custom',
    path_or_model=weights)

def read_image(path):
    return cv2.imread(path)[:, :, ::-1]

def inference(model, image, size=640):
    return model(image, size=size)

if __name__ == "__main__":
    image_path = "../images/2_348.png"
    model_path = './weights/last.pt'
    labels = ['JCB', 'Person', 'Truck', 'Helmet', 'Crane', 'Jacket']

    model = load_model(model_path)
    image = read_image(image_path)

    results = inference(model, image)
    dets = results.xyxy[0]
    dets = [list(map(int, lst)) for lst in dets]
    print(dets)