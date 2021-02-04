import cv2
from yolov5 import load_model, read_image, inference


def get_center(x1, y1, x2, y2):
    return ((x1+x2)//2), ((y1+y2)//2)

if __name__ == "__main__":
    image_path = "./images/2_348.png"
    model_path = './yolov5/weights/last.pt'
    labels = ['JCB', 'Person', 'Truck', 'Helmet', 'Crane', 'Jacket']
    colors = [(255, 0, 0), (0, 0, 255), (255, 0, 0), (255, 163, 255), (255, 0, 0), (0, 180, 255)]

    model = load_model(model_path)
    image = read_image(image_path)

    results = inference(model, image)
    dets = results.xyxy[0]
    dets = [list(map(int, lst)) for lst in dets]

    persons, helmets, jackets = [], [], []
    image = image[:, :, ::-1]
    for det in dets:
        if det[5] == 1:
            persons.append(det)
        elif det[5] == 3:
            helmets.append(det)
        elif det[5] == 5:
            jackets.append(det)
        else: 
            cv2.rectangle(image, (det[0], det[1]), (det[2], det[3]), colors[det[5]], thickness=2)
            cv2.putText(image, labels[det[5]], (det[0], det[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=colors[det[5]], thickness=2)
    
    flag_h = 0
    flag_j = 0
    for person in persons:
        x1, y1, x2, y2 = person[0], person[1], person[2], person[3]
        for helmet in helmets:
            h_cx, h_cy = get_center(helmet[0], helmet[1], helmet[2], helmet[3])
            if x1 <= h_cx <= x2 and y1 <= h_cy <= y2:
                flag_h = 1
        
        for jacket in jackets:
            j_cx, j_cy = get_center(jacket[0], jacket[1], jacket[2], jacket[3])
            if x1 <= j_cx <= x2 and y1 <= j_cy <= y2:
                flag_j = 1

        if flag_h and flag_j:
            cv2.rectangle(image, (person[0], person[1]), (person[2], person[3]), (0, 255, 0), thickness=2)
            cv2.putText(image, labels[person[5]], (person[0], person[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=2)
        else:
            cv2.rectangle(image, (person[0], person[1]), (person[2], person[3]), (0, 0, 255), thickness=2)
            cv2.putText(image, labels[person[5]], (person[0], person[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)

        flag_h = flag_j = 0
    cv2.imwrite("results.jpg", image)