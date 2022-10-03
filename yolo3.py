# YOLO object detection
import cv2 as cv
import numpy as np
import time
import PIL
from PIL import ImageFont, ImageDraw, Image
WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None

# Load names of classes and get random colors
classes = open('cfg/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('cfg/yolov3-spp.cfg', 'cfg/yolov3-spp.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# determine the output layer
ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

def white(image ):
    raw, column, c = image.shape
    width = raw
    height = column
    print(width, height)
    new = PIL.Image.new('RGB', size = (height, width), color = (0, 0, 0))
    #new.show()
    # 創建Draw對象:
    draw = ImageDraw.Draw(new)
    draw.line(xy=((20,20),(360,20)), fill=(255, 255, 255))
    draw.line(xy=((20,180),(360,180)), fill=(255, 255, 255))
    draw.line(xy=((20,20),(20,180)), fill=(255, 255, 255))
    draw.line(xy=((105,20),(105,180)), fill=(255, 255, 255))
    draw.line(xy=((190,20),(190,180)), fill=(255, 255, 255))
    draw.line(xy=((275,20),(275,180)), fill=(255, 255, 255))
    draw.line(xy=((360,20),(360,180)), fill=(255, 255, 255))



    draw.line(xy=((20,390),(360,390)), fill=(255, 255, 255))
    draw.line(xy=((20,560),(360,560)), fill=(255, 255, 255))
    draw.line(xy=((20,390),(20,560)), fill=(255, 255, 255))
    draw.line(xy=((105,390),(105,560)), fill=(255, 255, 255))
    draw.line(xy=((190,390),(190,560)), fill=(255, 255, 255))
    draw.line(xy=((275,390),(275,560)), fill=(255, 255, 255))
    draw.line(xy=((360,390),(360,560)), fill=(255, 255, 255))


                    #draw.
    return new


def load_image(image):
    global img, img0, outputs, ln
    img0 = image
    img = img0.copy()

    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time() - t0

    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    outputs = np.vstack(outputs)
    # print(path)
    post_process(img, outputs, 0.1)
    #cv.imshow('window', img)
    #cv.displayOverlay('window', f'forward propagation time={t:.3}')
    #cv.waitKey(100000)
    # cv.destroyAllWindows()


def post_process(img, outputs, conf):
    H, W = img.shape[:2]
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w // 2), int(y - h // 2)
            p1 = int(x + w // 2), int(y + h // 2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)

    if len(indices) > 0:
        newimg = white(img)
        draw = ImageDraw.Draw(newimg)

        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            print('(', x, ',', y, ')', x, y, classes[classIDs[i]])
            cv.putText(img, 'car', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print('x,y ', x+w/2,y+h/2)
            if not(180<y+h/2<390):
                draw.ellipse((x+w/2,y+h/2, x+w/2+10,y+h/2+10), fill='blue', outline='blue')
        #cv.imshow('test',img)
        #cv.waitKey(1000)
        #newimg.show()
        im_numpy = np.array(newimg)
        opencv_image=cv.cvtColor(im_numpy, cv.COLOR_RGB2BGR)
        #cv.imshow('test2',opencv_image)
        #cv.waitKey(1000)
        #new.imshow('test2',img)
        final = np.concatenate([img, opencv_image], axis=1)
        cv.imshow('final',final)
        cv.waitKey(1000)

def trackbar(x):
    global img
    conf = x / 100
    # img = img0.copy()
    post_process(img, outputs, conf)
    cv.displayOverlay('window', f'confidence level={conf}')
    cv.imshow('window', img)


cv.namedWindow('window')

cap = cv.VideoCapture('video/car4.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    #print(frame)
    print('loading ... ')
    white(frame)
    #cv.waitKey(100000)
    load_image(frame)
# paths = ['images/car1.jpg', 'images/car2.jpg', 'images/car3.jpg']

# for path in paths:
#    image = cv.imread(path)
#    load_image(image)
# image1 = cv.imread(path)
# image2 = cv.imread(path)
# image3 = cv.imread(path)
# load_image('images/car2.jpg')
# load_image('images/car1.jpg')
# load_image('images/car3.jpg')

