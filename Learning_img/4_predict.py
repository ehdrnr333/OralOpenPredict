import cv2
import dlib
from imutils import face_utils
import os
import tensorflow as tf

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def range_points(x,y,w,h):
    if x > h or x < 0 or y > w or y < 0:
        return False
    return True

def nomalize_data(data):
    if data is not -1:
        vect = [[], []]
        vect[0] = data[0][48:67]
        vect[1] = data[1][48:67]
        min_x = min(vect[0])
        min_y = min(vect[1])
        size = data[2][3]
        for k in range(0, 19):
            vect[0][k] = int((vect[0][k] - min_x) * 100 / size)
            vect[1][k] = int((vect[1][k] - min_y) * 100 / size)
        mid_x = 50 - vect[0][52 - 49]+1
        mid_y = 50 - vect[1][49 - 49]+1

        for k2 in range(0, 19):
            vect[0][k2] = vect[0][k2] + mid_x
            vect[1][k2] = vect[1][k2] + mid_y
    return vect

def numeric_res(data):
    if data is -1:
        return "error"
    vect = nomalize_data(data)
    if (vect[1][67-49]-vect[1][63-49]) > 3:
        return "open"
    else:
        return "close"

def predict_video(mean):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)
        data = [[], [], [0,0,0,0]]
        try:
            rects[0]
        except IndexError as e:
            pass

        else:
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            data[2][0] = x
            data[2][1] = y
            data[2][2] = w
            data[2][3] = h
            for (x, y) in shape:
                data[0].append(int(x))
                data[1].append(int(y))
            if mean is 1:
                res = numeric_res(data)
            print(res)
        cv2.imshow("Output", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def image_points(img_path):
    if os.path.exists(img_path) is False:
        return -1
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    data = [[],[],[0,0,0,0]]
    count = 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        data[2][0] = x
        data[2][1] = y
        data[2][2] = w
        data[2][3] = h
        if count>0:
            break
        for (x, y) in shape:
            data[0].append(int(x))
            data[1].append(int(y))
        count = count+1
    return data

def predict_photo(mean):
    load_image_open = "./eval/open/"
    load_image_close = "./eval/close/"
    i = 1
    while True:
        img_path = load_image_open + "/" + str(i) + "_open.jpg"
        if os.path.exists(img_path) is False:
            break
        res = image_points(img_path)
        if mean is 1 and res[2][0] is not 0:
            res = numeric_res(res)
        print(res)
        i = i+1
    i = 1
    while True:
        img_path = load_image_close + "/" + str(i) + "_close.jpg"
        if os.path.exists(img_path) is False:
            break
        res = image_points(img_path)
        if mean is 1 and res[2][0] is not 0:
            res = numeric_res(res)
        print(res)
        i = i+1

if __name__ == '__main__':
    predict_photo(1)