import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2 as cv
import os
from sklearn.utils import shuffle
import pickle
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

#%matplotlib inline
def _sep():
    print('***********************************************************************************')

def take_img(ruta):
    vs =  cv.VideoCapture(0)
    key = cv.waitKey(1)
    while True:
        try:
            check, frame = vs.read()
            #print(check) #prints true as long as the webcam is running
            #print(frame) #prints matrix values of each framecd
            cv.imshow("Capturando", frame)
            key = cv.waitKey(1)
            if key == ord('s'):
                print('>> Capturando imagen...')
                cv.imwrite(filename=ruta, img=frame)
                vs.release()
                img_new = cv.imread(ruta,3)
                img_new = cv.imshow("Imgen tomada", img_new)
                cv.waitKey(1650)
                cv.destroyAllWindows()
                print(">>> Procesando Imagen...")
                img_readed = cv.imread(ruta, cv.IMREAD_ANYCOLOR)
                im = cv.getRectSubPix(img_readed, (360, 360), (320, 240))
                #im = img_readed[400:400, 200:200]
                img_resized = cv.resize(im,(64,64))
                img_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
                img_rgb = img_rgb[...,::-1].copy()
                img_final = cv.imwrite(filename=ruta, img=img_rgb)
                print(">>>> Imagen guardada!, en la ruta: {}".format(ruta))
                break
        except(KeyboardInterrupt):
            print("apagando camera.")
            vs.release()
            print("Camera fuera.")
            cv.destroyAllWindows()
            break

def charge_dataset(ruta):
    File = open(ruta, 'rb')
    data_x_tmp = pickle.load(File)
    data_y = pickle.load(File)
    File.close()
    del(File)
    return data_x_tmp, data_y

def convpool(X, W, b):
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)

def _print_img(dimx):
    print("Dimensión de la imagen {}".format(dimx.shape))

def get_images(test_ruta, Xt):
    img = cv.imread(test_ruta,3)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ima = cv.resize(img,(64,64))
    Xt.append(ima)
    return Xt

def show_img(ruta):
    #img_reshape = img.reshape([64,64,3])
    #m2 = im2.T;
    #plt.imshow(img_reshape)
    img_new = cv.imread(ruta,3)
    img_new = cv.imshow("Imgen tomada", img_new)
    cv.waitKey(2000)
    cv.destroyAllWindows()

def cr_res():
    res = ([[0], [1]])
    #print(res)
    return res

def OHE(res):
    oneHot = OneHotEncoder()
    oneHot.fit(res)
    res = oneHot.transform(res).toarray()
    return res

def create_hipotesis(img_resh):
    n_char = np.shape(img_resh)[1]         # Número de cararcteristicas
    n_class = 2
    hidden_layer_size = 300
    samples = np.shape(img_resh)[0]
    colmns = np.shape(img_resh)[1]
    #Placeholders
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [samples, 64, 64, 3])
    Y = tf.placeholder(tf.float32, [2, n_class])
    W1 = tf.get_variable("W1", [8, 8, 3, 32])
    b1 = tf.get_variable("b1", 32)
    W2 = tf.get_variable("W2",[8, 8, 32, 64])
    b2 = tf.get_variable("b2", 64)
    W3 = tf.get_variable("W3",[16*16*64, hidden_layer_size])
    b3 = tf.get_variable("b3",[hidden_layer_size])
    W4 = tf.get_variable("W4",[hidden_layer_size, n_class])
    b4 = tf.get_variable("b4",n_class)
    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu( tf.matmul(Z2r, W3) + b3 )
    Z = tf.matmul(Z3, W4) + b4
    y_ = tf.nn.tanh(Z)
    return X, Y, y_

def prediction_model(img, predice):
    img_reshape = img.reshape([64,64,3])
    #m2 = im2.T;
    plt.imshow(img_reshape)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    if(predice[0] == 1):
        #if(predice[0] == 1):
        response = 'Hombre'
    elif(predice[0] == 0):
        response = 'Mujer'
    else:
        response = 'Indeterminado'
    _sep()
    print('>>>> Lo predice como: {}'.format(response))
    _sep()

def _title():
    os.system ("clear")
    print("<<< -Clasificador Hombre/Mujer- >>>\n")


def main(ruta):
    while True:
        _title()
        take_img(ruta)
        test_ruta = ruta
        Xt = get_images(test_ruta, Xt=list())
        img = np.array(Xt)
        #_print_img(img)
        show_img(ruta)
        img_resh = img.reshape(img.shape[0],(img.shape[1]*img.shape[2]*img.shape[3]))
        res = OHE(cr_res())

        #model prediction
        X, Y, y_ = create_hipotesis(img_resh)
        predice = tf.argmax(y_, 1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, os.getcwd()+"/modeloCNN.ckpt")
            img_resh = img_resh.reshape(img_resh.shape[0], 64, 64, 3)
            predice = sess.run([predice], feed_dict={X: img_resh})

        prediction_model(img, predice)
        print('Desea intentarlo nuevamente [S/N]??')
        again = str(input(': '))
        if(again.lower() == 's'.lower()):
            os.system ("clear")
        else:
            print('\n>> Cerrando programa...\n Adios!!!')
            break


if __name__ == '__main__':
    ruta = 'images/test/test19.jpg'
    main(ruta)
