from collections import OrderedDict
from imutils import face_utils
import dlib
import cv2
import os

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def range_points(x,y,w,h):
    if x > h or x < 0 or y > w or y < 0:
        return False
    return True

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def image_points(img_path, point_path):
    if os.path.exists(img_path) is False:
        return -1

    image = cv2.imread(img_path)
    width, height = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    points = []
    f = open(point_path, 'w')
    xywh = ""
    count = 0
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        xywh= ""+str(x)+' '+str(y)+' '+str(w)+' '+str(h)
        if count>0:
            break
        for (x, y) in shape:
            if range_points(x,y,width,height)is False:
                return 0

            points.append((int(x), int(y)))
            data = "%d %d\n" %(x,y)
            f.write(data)
        count = count+1
    f.write(xywh)
    f.close()
    return 1

def keyword_process(keyword):
    load_image = "./origin/" + keyword
    save_points = "./origin_text/" + keyword
    i = 1
    while True:
        img_path = load_image + "/" + str(i) + "_" + keyword + ".jpg"
        point_path = save_points + "/" + str(i) + "_" + keyword + ".txt"
        res = image_points(img_path, point_path)
        if res is -1:
            break
        print(">>point:" + keyword + " " + str(i) + " save")
        i = i+1
    print(":::point:"+keyword+"finish")

def keyword_advanced(keyword):
    i = 0
    while True:
        i = i+1
        point_path = "./origin_text/" + keyword + "/" + str(i) + "_" + keyword + ".txt"
        save_path =  "./tensor_text/" + keyword + "/" + str(i) + "_" + keyword + ".txt"
        if os.path.exists(point_path) is False:
            break
        fw = open(save_path, 'w')
        data = read_txt(point_path)

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
            print(vect)

            for k2 in range(0, 19):
                vect[0][k2] = vect[0][k2] + mid_x
                vect[1][k2] = vect[1][k2] + mid_y
                f_str = "%d %d\n" % (vect[0][k2], vect[1][k2])
                fw.write(f_str)
            fw.close()

        print(">>point:" + keyword + " " + str(i) + " save")

def read_txt(point_path):
    data = [[],[],[0,0,0,0]]
    f = open(point_path, 'r')

    for i in range(0, 68):
        line = f.readline()
        if line is "": return -1
        pt = line.split(' ')
        data[0].append(int(pt[0]))
        data[1].append(int(pt[1]))
    line = f.readline()
    pt = line.split(' ')
    data[2][0] = int(pt[0])
    data[2][1] = int(pt[1])
    data[2][2] = int(pt[2])
    data[2][3] = int(pt[3])
    return data

if __name__ == '__main__':
    #keyword_process("open")
    #keyword_process("close")
    keyword_advanced("open")
    keyword_advanced("close")