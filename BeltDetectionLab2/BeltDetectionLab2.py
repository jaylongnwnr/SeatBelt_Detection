import cv2
import numpy as np
import logging
import time
from contextlib import contextmanager
from os.path import dirname, abspath

VIDEO = "test.mp4"
WEIGHTS = "YOLOFI2.weights"
CONFIG = "YOLOFI.cfg"
OBJ_NAMES = "obj.names"
SAVE_PATH = dirname(dirname(abspath(__file__))) + "/"

logging.basicConfig(level=logging.INFO)

@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 0.3, theta, 9.0, 0.6, 50, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        accum = np.maximum(accum, fimg)
    return accum

def main():
    net = cv2.dnn.readNet(WEIGHTS, CONFIG)

    layers_names = net.getLayerNames()
    outs = net.getUnconnectedOutLayers()
    if len(outs.shape) == 2:
        output_layers = [layers_names[i[0] - 1] for i in outs]
    else:
        output_layers = [layers_names[i - 1] for i in outs]
    print("Output layers:", output_layers)

    with video_capture(VIDEO) as cap:
        frame_id = 0
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, channels = frame.shape

            # 影像前處理
            clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
            R, G, B = cv2.split(frame)
            output1_R = clahe.apply(R)
            output1_G = clahe.apply(G)
            output1_B = clahe.apply(B)
            frame = cv2.merge((output1_R, output1_G, output1_B))
            frame = cv2.fastNlMeansDenoisingColored(frame, None, h=3, hColor=5, templateWindowSize=7, searchWindowSize=21)

            filters = build_filters()
            frame = process(frame, filters)

            # 建立輸入 blob
            blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(480, 480), mean=(0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)

            outs = net.forward(output_layers)

            beltcornerdetected = False
            beltdetected = False

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.2:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if class_id == 1:
                            beltcornerdetected = True
                        elif class_id == 0:
                            beltdetected = True

            # 用 cv2.putText 顯示文字在畫面左上角
            text = f"Frame {count}: Belt={beltdetected}, Corner Belt={beltcornerdetected}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            count += 1

            cv2.imshow("Detection", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
