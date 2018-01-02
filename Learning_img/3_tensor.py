# Lab 9 XOR
import tensorflow as tf
import numpy as np
import os
import cv2
import dlib
from imutils import face_utils

def read_txt(point_path):
    data = []
    f = open(point_path, 'r')

    for i in range(0, 19):
        line = f.readline()
        if line is "": return -1
        pt = line.split(' ')
        data.append(int(pt[0]))
        data.append(int(pt[1]))
    f.close()
    return data

def make_XY():
    num_0 = "./tensor_text/open/"
    num_1 =  "./tensor_text/close/"
    x_data = []
    y_data = []

    for filename in os.listdir(num_0):
        data = read_txt(num_0+filename)
        if data is not -1:
            x_data.append(data)
            y_data.append([0,1])
    for filename in os.listdir(num_1):
        data = read_txt(num_1+filename)
        if data is not -1:
            x_data.append(data)
            y_data.append([1,0])
    return x_data, y_data

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def train_data2():
    x_data, y_data = make_XY()
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    with tf.name_scope('layer1'):
        W1 = tf.Variable(tf.random_uniform([38, 10], -1., 1.), name='W1')
        L1 = tf.nn.relu(tf.matmul(X, W1))

        tf.summary.histogram("X", X)
        tf.summary.histogram("Weights", W1)

    with tf.name_scope('layer2'):
        W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
        L2 = tf.nn.relu(tf.matmul(L1, W2))

        tf.summary.histogram("Weights", W2)

    with tf.name_scope('output'):
        W3 = tf.Variable(tf.random_uniform([20, 2], -1., 1.), name='W3')
        model = tf.matmul(L2, W3)

        tf.summary.histogram("Weights", W3)
        tf.summary.histogram("Model", model)

    with tf.name_scope('optimizer'):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(cost, global_step=global_step)

        tf.summary.scalar('cost', cost)

    #########
    # 신경망 모델 학습
    ######
    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    for step in range(100):
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})

        print('Step: %d, ' % sess.run(global_step),
              'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

        summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=sess.run(global_step))

    saver.save(sess, './model/dnn.ckpt', global_step=global_step)

    #########
    # 결과 확인
    ######
    prediction = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)
    print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
    print('실제값:', sess.run(target, feed_dict={Y: y_data}))

    is_correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
    writer.close()

    text_photo = open("photo_res.txt", 'w')

    load_image_open = "./eval/open/"
    load_image_close = "./eval/close/"
    i = 1
    while True:
        img_path = load_image_open + "/" + str(i) + "_open.jpg"
        if os.path.exists(img_path) is False:
            break
        res = image_points(img_path)

        if res[2][0] is not 0:
            num_res = numeric_res(res)
            res = nomalize_data(res)
            res2 = [[]]
            for c in range(0, 19):
                res2[0].append(res[0][c])
                res2[0].append(res[1][c])
            x_data = np.array(res2, dtype=np.float32)
            fin = sess.run(prediction, feed_dict={X: x_data})
            write_str = "predict:%d:::%d:::>>>1\n" %(fin, num_res)
            text_photo.write(write_str)
        i = i+1
    i = 1
    while True:
        img_path = load_image_close + "/" + str(i) + "_close.jpg"
        if os.path.exists(img_path) is False:
            break
        res = image_points(img_path)

        if res[2][0] is not 0:
            num_res = numeric_res(res)
            res = nomalize_data(res)
            res2 = [[]]
            for c in range(0, 19):
                res2[0].append(res[0][c])
                res2[0].append(res[1][c])
            x_data = np.array(res2, dtype=np.float32)
            fin = sess.run(prediction, feed_dict={X: x_data})
            write_str = "predict:%d:::%d:::>>>0\n" %(fin, num_res)
            text_photo.write(write_str)
        i = i+1
    text_photo.close()
    text_video = open("video_res.txt", 'w')
    ################################################################
    cap = cv2.VideoCapture(0)
    mystr = ""
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
            res = data
            num_res = numeric_res(res)
            res = nomalize_data(res)
            res2 = [[]]
            for c in range(0, 19):
                res2[0].append(res[0][c])
                res2[0].append(res[1][c])
            x_data = np.array(res2, dtype=np.float32)
            fin = sess.run(prediction, feed_dict={X: x_data})
            afg="predict:%d:::%d:::\n" % (fin, num_res)
            mystr = mystr+"predict:%d:::%d:::\n" % (fin, num_res)

        cv2.imshow("Output", frame)
        print(afg)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    print(mystr)
    text_video.write(mystr)
    print(mystr)
    text_video.close()
    cap.release()
    cv2.destroyAllWindows()





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
        return 1
    else:
        return 0

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


if __name__ == '__main__':
    train_data2()


