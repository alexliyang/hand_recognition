import os
import numpy as np
import tensorflow as tf
from sklearn import model_selection
import utils
from PIL import Image
import random
import dill

# 首先要读取图片信息并导入。
PATH = './test' # 所有的手势图片都放在里面
img_rows = 200
img_cols = 200
img_channels = 1
batch_size = 32
nb_classes = 4 # 类别


def modlist(path):
    # 列出path里面所有文件信息
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist

def Initializer():
    # 初始化数据，产生训练测试数据和标签
    imlist = modlist(PATH)
    total_images = len(imlist) # 样本数量
    immatrix = np.array([np.array(Image.open(PATH+'/'+image).convert('L')).flatten() for image in imlist], dtype='float32')
    # 注 PIL 中图像共有9中模式 模式“L”为灰色图像 0黑 255白
    # 转换公式 L = R * 299/1000 + G * 587/1000+ B * 114/1000
    # 开始创建标签
    label = np.ones((total_images, ), dtype=int)
    samples_per_class = total_images / nb_classes # 每类样本的数量，（由于录制的时候录制的一样多，这里可以这样写，如果不一样多，标签就需要根据文件名来进行获取）
    s = 0
    r = samples_per_class
    # 开始赋予标签（01234）
    for index in range(nb_classes):
        # 0-300: 0
        # 301-601:1
        #...
        label[int(s):int(r)] = index
        s = r
        r = s + samples_per_class
    print(label)
    #data, label = utils.shuffle(immatrix, label, randomstate=2)
    data = immatrix
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, label, test_size=0.2, random_state=4)
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels) #  tensorflow的图像格式为[batch, W, H, C]
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels) #  tensorflow的图像格式为[batch, W, H, C]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    return X_train, X_test, y_train, y_test

def get_batch(X, y, batch_size):
    # tensorflow 利用batch的思想来加快训练速度
    data = []
    label = []
    m = X.shape[0]
    for _ in range(batch_size):
        index = random.randrange(m) # 随机选择一个整数
        data.append(X[index])
        tmp = np.zeros(NUM_LABELS, dtype=np.float32)
        tmp[y[index]] = 1.0
        label.append(tmp)
    return np.array(data), np.array(label) # 输出为ndarray
# 测试数据
# a, b, c, d = Initializer()
# print(a.shape) # 1204*200*200*1  每一行一个样本 (训练样本)
# print(b.shape) # 301*200*200*1
# print(c)
# print(d)




##################搭建网络#########################3333
INPUT_NODE = img_rows * img_cols
OUTPUT_NODE = 4

Image_size = 20
NUM_LABELS = 4

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE= 4

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE= 3

# 第一层卷积层的尺寸和深度
CONV3_DEEP = 64
CONV3_SIZE= 4

# 第二层卷积层的尺寸和深度
CONV4_DEEP = 64
CONV4_SIZE= 4

FC_SIZE1 = 512
FC_SIZE2 = 128

# 训练用参数
REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 1000
MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.ckpt'

def get_batch(X, y, batch_size):
    data = []
    label = []
    m = X.shape[0]
    for _ in range(batch_size):
        index = random.randrange(m)
        data.append(X[index])
        tmp = np.zeros(NUM_LABELS, dtype=np.float32)
        tmp[y[index]] = 1.0
        label.append(tmp)
    return np.array(data), np.array(label)


# 定义前向卷积 添加：dropout 训练有 测试没有
def inference(input_tensor, train, regularizer):
    with tf.name_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, img_channels, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="VALID") # 196*196*32
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding="SAME") # 98*98*32

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1,1,1,1], padding="VALID") # 96*96*64
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") # 48*48*64

    with tf.variable_scope('layer5-conv3'):
        conv3_weight = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias', [CONV3_DEEP], initializer=tf.constant_initializer(0.0))

        conv3 = tf.nn.conv2d(pool2, conv3_weight, strides=[1,1,1,1], padding="VALID") # 44*44*64
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 22*22*64

    with tf.variable_scope('layer7-conv4'):
        conv4_weight = tf.get_variable("weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('bias', [CONV4_DEEP], initializer=tf.constant_initializer(0.0))

        conv4 = tf.nn.conv2d(pool3, conv4_weight, strides=[1, 1, 1, 1], padding="VALID")  # 18*18*64
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope('layer8-pool4'):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 9*9*64

    # 然后将第8层的输出变为第9层输入的格式。 后面全连接层需要输入的是向量 将矩阵拉成一个向量
    pool_shape = pool4.get_shape().as_list()
    # pool_shape[0]为一个batch中数据个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool4, [pool_shape[0], nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE1], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE1, FC_SIZE2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [FC_SIZE2], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [FC_SIZE2, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit # 注 ： 这里没有经过softmax，后面在计算cross_entropy时候利用内置的函数会计算。


####################训练##########################3
def train(X_train, y_train):
    x = tf.placeholder(tf.float32, [batch_size, img_rows, img_cols, img_channels], name='x-input')
    y = tf.placeholder(tf.float32, [batch_size, OUTPUT_NODE], name='y-input')

    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 前向传播
    y_ = inference(x, train=True, regularizer=regularizer) # 预测值
    global_step = tf.Variable(0, trainable=False) # 不可训练

    #定义损失函数
    # 滑动平均
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AAVERAGE_DECAY, global_step)
    # variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(y, 1), logits=y_) #
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses')) # 计算总loss

    # learninig_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 1204//batch_szie, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss, global_step=global_step)

    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name='train')
    # train_op = tf.group(train_step, variable_averages_op)

    # 保存模型
    saver = tf.train.Saver()
    pointer = 0
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(TRAINING_STEPS):
            xs, ys = get_batch(X_train, y_train, batch_size=batch_size)
            # xs, ys = get_next_batch(X_train, y_train, batch_size=batch_szie)
            # ys = tf.reshape(tf.one_hot(ys, depth=5), [batch_szie, OUTPUT_NODE])
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x: xs, y: ys})
            print("----------------------", i, " : ", loss_value,  "-------------------------------")
            #if step % 1000 == 0:
                #print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step) # 保存模型。


X_train, X_test, y_train, y_test = Initializer()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
'''train(X_train,y_train)
dill.dump_session('1.pkl')
'''
dill.load_session('1.pkl')
def test(X_test, y_test):
    # EVAL_INTERVAL_SECS = 10 # 每10秒加载一次模型，并在测试数据上测试准确率
    with tf.Graph().as_default() as g: # 设置默认graph
        # 定义输入输出格式
        #
        x = tf.placeholder(tf.float32, [None, img_rows, img_cols, img_channels], name='x-input')
        y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

        y_ = inference(x, train=None, regularizer=None) # 测试时 不关注正则化损失的值

        # 开始计算正确率
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 加载模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state会自动找到目录中的最新模型文件名
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 得到迭代轮数
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # model.ckpt-3000
                for _ in range(X_test.shape[0]):
                    xs, ys = get_batch(X_test, y_test, batch_size=1) # 测试用
                    # print(ys.shape)
                    label, accuracy_score = sess.run([y_, accuracy], feed_dict={x: xs, y: ys})
                    print("实际手势： %s，  预测手势： %s" % (output[np.argmax(ys)], output[np.argmax(label)]))
                    # print("After %s training steps(s), test accuracy = %f" % (global_step, accuracy_score))

            else:
                print("No checkpoint, Training Firstly.")
                return













def Gussgesture(X_test):
    # EVAL_INTERVAL_SECS = 10 # 每10秒加载一次模型，并在测试数据上测试准确率
    with tf.Graph().as_default() as g: # 设置默认graph
        # 定义输入输出格式
        #
        x = tf.placeholder(tf.float32, [1, img_rows, img_cols, img_channels], name='x-input')
        y_ = inference(x, train=None, regularizer=None) # 测试时 不关注正则化损失的值

        # 开始计算正确率
        # correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print(X_test.shape)
        # 加载模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state会自动找到目录中的最新模型文件名
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 得到迭代轮数
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # model.ckpt-3000
                # print(ys.shape)
                label = sess.run(y_, feed_dict={x: X_test.reshape(1, X_test.shape[0], X_test.shape[1], X_test.shape[2])})
                #print(label)
                if np.argmax(label) == 0:
                    print("预测手势： %s" % '放大')
                if np.argmax(label) == 1:
                    print("预测手势： %s" % '缩小')
                if np.argmax(label) == 2:
                    print("预测手势： %s" % '向右')
                if np.argmax(label) == 3:
                    print("预测手势： %s" % '向左')
                # PLOT(label)
                # print("After %s training steps(s), test accuracy = %f" % (global_step, accuracy_score))
                return np.argmax(label)
            else:
                print("No checkpoint, Training Firstly.")
                return
# 导入需要的包
import cv2
import os
import time
# 设置一些常用的一些参数
# 显示的字体 大小 初始位置等
font = cv2.FONT_HERSHEY_SIMPLEX #  正常大小无衬线字体
size = 0.5
fx = 10
fy = 355
fh = 18
# ROI框的显示位置
x0 = 300
y0 = 100
# 录制的手势图片大小
width = 200
height = 200
# 每个手势录制的样本数
numofsamples = 300
counter = 0 # 计数器，记录已经录制多少图片了
# 存储地址和初始文件夹名称
gesturename = ''
path = ''
# 标识符 bool类型用来表示某些需要不断变化的状态
binaryMode = False # 是否将ROI显示为而至二值模式
saveImg = False # 是否需要保存图片

# 创建一个视频捕捉对象
cap = cv2.VideoCapture(0) # 0为（笔记本）内置摄像头
# 显示ROI为二值模式
# 图像的二值化，就是将图像上的像素点的灰度值设置为0或255，
# 也就是将整个图像呈现出明显的只有黑和白的视觉效果。

#  cv2.threshold  进行阈值化
# 第一个参数  src     指原图像，原图像应该是灰度图
# 第二个参数  x     指用来对像素值进行分类的阈值。
# 第三个参数    y  指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
# 有两个返回值 第一个返回值（得到图像的阈值）   二个返回值 也就是阈值处理后的图像

def binaryMask(frame, x0, y0, width, height):
    # 显示方框
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0))
    #提取ROI像素
    roi = frame[y0:y0+height, x0:x0+width] #
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 高斯模糊 斯模糊本质上是低通滤波器，输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和
    # 高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大
    blur = cv2.GaussianBlur(gray, (5, 5), 2) # 高斯模糊，给出高斯模糊矩阵和标准差

    # 当同一幅图像上的不同部分的具有不同亮度时。这种情况下我们需要采用自适应阈值
    # 参数： src 指原图像，原图像应该是灰度图。 x ：指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
    #  adaptive_method  指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
    # block_size           指用来计算阈值的象素邻域大小: 3, 5, 7, ..
    #   param1           指与方法有关的参数    #
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # ret还是bool类型
    # 保存手势
    if saveImg == True and binaryMode == True:
        saveROI(res)
    elif saveImg == True and binaryMode == False:
        saveROI(roi)
    """这里可以插入代码调用网络"""

    return res

# 保存ROI图像
def saveROI(img):
    global path, counter, gesturename, saveImg
    if counter > numofsamples:
        # 恢复到初始值，以便后面继续录制手势
        saveImg = False
        gesturename = ''
        counter = 0
        return

    counter += 1
    name = gesturename + str(counter) # 给录制的手势命名
    print("Saving img: ", name)
    cv2.imwrite(path+name+'.png', img) # 写入文件
    time.sleep(0.05)

def skinal(img):
    # load an original image
    ################################################################################
    # better way: using numpy


    # warning: we can only change pixels in gray or single-channel image

    # modify green value: (row,col,channel)
    img.itemset((100, 200, 1), 255)


    rows, cols, channels = img.shape


    # copy original image
    imgSkin = img.copy()

    for r in range(rows):
        for c in range(cols):

            # get pixel value
            B = img.item(r, c, 0)
            G = img.item(r, c, 1)
            R = img.item(r, c, 2)

            # non-skin area if skin equals 0, skin area otherwise
            skin = 0

            if (abs(R - G) > 15) and (R > G) and (R > B):
                if (R > 95) and (G > 40) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                    skin = 1
                    # print 'Condition 1 satisfied!'
            elif (R > 220) and (G > 210) and (B > 170):
                if (abs(R - G) <= 15) and (R > G) and (R > B):
                    skin = 1
                    # print 'Condition 2 satisfied!'

            if 0 == skin:
                imgSkin.itemset((r, c, 0), 0)
                imgSkin.itemset((r, c, 1), 0)
                imgSkin.itemset((r, c, 2), 0)
                # print 'Skin detected!'

    # convert color space of images because of the display difference between cv2 and matplotlib

    imgSkin = cv2.cvtColor(imgSkin, cv2.COLOR_BGR2RGB)

    # display original image and skin image

    ################################################################################
    return imgSkin

while(True):
    # 读帧
    ret, frame = cap.read() # 返回的第一个参数为bool类型，用来表示是否读取到帧，如果为False说明已经读到最后一帧。frame为读取到的帧图片
    # 图像翻转（如果没有这一步，视频显示的刚好和我们左右对称）
    frame = cv2.flip(frame, 2)# 第二个参数大于0：就表示是沿y轴翻转
    frame = skinal(frame)
    # 显示ROI区域 # 调用函数
    roi = binaryMask(frame, x0, y0, width, height)

    # 显示提示语
    #cv2.putText(frame, "Option: ", (fx, fy), font, size, (0, 255, 0))  # 标注字体
    #cv2.putText(frame, "b-'Binary mode'/ r- 'RGB mode' ", (fx, fy + fh), font, size, (0, 255, 0))  # 标注字体
    #cv2.putText(frame, "p-'prediction mode'", (fx, fy + 2 * fh), font, size, (0, 255, 0))  # 标注字体
    #cv2.putText(frame, "s-'new gestures(twice)'", (fx, fy + 3 * fh), font, size, (0, 255, 0))  # 标注字体
    #cv2.putText(frame, "q-'quit'", (fx, fy + 4 * fh), font, size, (0, 255, 0))  # 标注字体


    Roi = np.reshape(roi, [width, height, 1])
    # print(Roi.shape)
    Gussgesture(Roi)
    #gesture_copy = gesture


    key = cv2.waitKey(1) & 0xFF # 等待键盘输入，
    if key == ord('b'):  # 将ROI显示为二值模式
       # binaryMode = not binaryMode
       binaryMode = True
       print("Binary Threshold filter active")
    elif key == ord('r'): # RGB模式
        binaryMode = False

        if key == ord('i'):  # 调整ROI框
            y0 = y0 - 5
    elif key == ord('k'):
        y0 = y0 + 5
    elif key == ord('j'):
        x0 = x0 - 5
    elif key == ord('l'):
        x0 = x0 + 5

    if key == ord('p'):
        """调用模型开始预测"""
        print("using CNN to predict")
        """调用模型开始预测, 对二值图像预测，所以要在二值函数里面调用，预测新采集的手势"""
        # print("Prediction Mode - {}".format(guessGesture))
        # Prediction(roi)
        Roi = np.reshape(roi, [width, height, 1])
        # print(Roi.shape)
        gesture = Gussgesture(Roi)
        #gesture_copy = gesture
    if key == ord('q'):
        break

    if key == ord('s'):
        """录制新的手势（训练集）"""
        # saveImg = not saveImg # True
        if gesturename != '':  #
            saveImg = True
        else:
            print("Enter a gesture group name first, by enter press 'n'! ")
            saveImg = False
    elif key == ord('n'):
        # 开始录制新手势
        # 首先输入文件夹名字
        gesturename = (input("enter the gesture folder name: "))
        os.makedirs(gesturename)

        path = "./" + gesturename + "/" # 生成文件夹的地址  用来存放录制的手势

    #展示处理之后的视频帧
    cv2.imshow('frame', frame)
    if (binaryMode):
        cv2.imshow('ROI', roi)
    else:
        cv2.imshow("ROI", frame[y0:y0+height, x0:x0+width])

#最后记得释放捕捉
cap.release()
cv2.destroyAllWindows()

