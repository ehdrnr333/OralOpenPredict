import os
import shutil
import cv2
import dlib
from imutils import face_utils
import numpy as np
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# <Draw Line>
# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList();
    size = img.shape[:2]
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def draw_graph(img, points):
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect);

    for p in points:
        subdiv.insert(p)

    draw_delaunay(img, subdiv, (255, 255, 255));
    return img

def rename_allfile(keyword):
    load_path = "./eval/" + keyword + "_temp/"
    save_path = "./eval/" + keyword + "/"
    i = 1
    for filename in os.listdir(load_path):
        shutil.copy(load_path+filename, save_path+str(i) + "_" + keyword + ".jpg")
        i = i+1

def rename_allfile2(keyword):
    load_path = "./origin/" + keyword + "_temp/"
    save_path = "./origin/" + keyword + "/"
    i = 1
    for filename in os.listdir(load_path):
        shutil.copy(load_path+filename, save_path+str(i) + "_" + keyword + ".jpg")
        i = i+1

def add_emotion_image():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)
    count = 1

    while True:
        points = []
        ret, image = cap.read()
        width, height = image.shape[:2]

        save_img = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            count = 1
            for (x, y) in shape:
                if x > height - 3:
                    x = height - 3
                elif x < 3:
                    x = 3
                if y > width - 3:
                    y = width - 3
                elif y < 3:
                    y = 3
                points.append((int(x), int(y)))
                if count == 60:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                count = count+1
        image = draw_graph(image, points)
        cv2.imshow("Output", image)
        k = cv2.waitKey(1) & 0xFF
        if k == 32: # space
            k2 = cv2.waitKey(0) & 0xFF
            print(k2)
            if k2 == 97:
                cv2.imwrite("./origin/open_temp/"+str(count)+"_open.jpg",save_img)
                count = count+1
                print(">> save open img")
            elif k2 == 104:
                cv2.imwrite("./origin/close_temp/"+str(count)+"_close.jpg",save_img)
                count = count + 1
                print(">> save close img")
        elif k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #rename_allfile("open")
    #rename_allfile("close")
    add_emotion_image()
    #rename_allfile2("open")
    #rename_allfile2("close")
