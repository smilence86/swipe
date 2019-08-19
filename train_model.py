import math, random, time as time, datetime, shutil, os, cv2
import face_recognition
import tensorflow as tf
import numpy as np
from tool import zoomIn


EPOCH=20                #所有样本循环训练次数
n_classes=10            #分类个数
width=200               #输入图片宽度
height=200              #输入图片高度
channels=3              #输入图片通道数（RGB）
batch_size = 10          #小批量梯度下降，1代表随机梯度下降
Learn_rate = 0.0002     #学习率
# 输入：width*height图片，前面的None是batch size
x = tf.compat.v1.placeholder(tf.float32, shape=[None, width, height, channels])
# 输出：n_classes个分类
y = tf.compat.v1.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.compat.v1.placeholder(tf.float32)
learn_rate = tf.compat.v1.placeholder(tf.float32)


def weight_variable(shape, std):
    initial = tf.random.truncated_normal(shape, mean=0.0, stddev=std)
    return tf.Variable(initial)

def bias_variable(shape, std):
    initial = tf.constant(std, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, b, s=1):
    x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    out_size = W.get_shape().as_list()[3]
    # print(out_size)
    x = batch_norm(x, out_size)
    return tf.nn.relu(x)

def maxPool(x, k=2):
    return tf.nn.max_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def avgPool(x, k=2):
    return tf.nn.avg_pool2d(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

weights = {
    #
    'wc1': weight_variable([3, 3, 3, 32], 0.1),
    #
    'wc2': weight_variable([3, 3, 32, 64], 0.1),
    #
    'wc3': weight_variable([3, 3, 64, 128], 0.1),
    #
    'wc4': weight_variable([3, 3, 128, 256], 0.1),
    #
    'wc5': weight_variable([3, 3, 256, 512], 0.1),
    #1×1卷积核，降低输出维度
    'wc6': weight_variable([1, 1, 512, 256], 0.1),
    # fully connected, 
    'w_fc1': weight_variable([4*4*256, 1024], 0.1),
    # fully connected, 
    'w_fc2': weight_variable([1024, 512], 0.1),
    # 
    'out': weight_variable([512, n_classes], 0.1)
}

biases = {
    'bc1': bias_variable([32], 0.1),
    'bc2': bias_variable([64], 0.1),
    'bc3': bias_variable([128], 0.1),
    'bc4': bias_variable([256], 0.1),
    'bc5': bias_variable([512], 0.1),
    'bc6': bias_variable([256], 0.1),
    'b_fc1': bias_variable([1024], 0.1),
    'b_fc2': bias_variable([512], 0.1),
    'out': bias_variable([n_classes], 0.1)
}

def batch_norm(x, n_out, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.compat.v1.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = mean_var_with_update()
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def conv_net(x, weights, biases, keep_prob):
    # x = tf.reshape(x, shape=[-1, 100, 100, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.shape)
    max_pool1 = maxPool(conv1, k=2)
    print(max_pool1.shape)

    conv2 = conv2d(max_pool1, weights['wc2'], biases['bc2'])
    print(conv2.shape)
    max_pool2 = maxPool(conv2, k=2)
    print(max_pool2.shape)

    conv3 = conv2d(max_pool2, weights['wc3'], biases['bc3'])
    print(conv3.shape)
    max_pool3 = maxPool(conv3, k=2)
    print(max_pool3.shape)

    conv4 = conv2d(max_pool3, weights['wc4'], biases['bc4'])
    print(conv4.shape)
    max_pool4 = maxPool(conv4, k=2)
    print(max_pool4.shape)

    conv5 = conv2d(max_pool4, weights['wc5'], biases['bc5'])
    print(conv5.shape)
    max_pool5 = maxPool(conv5, k=2)
    print(max_pool5.shape)

    conv6 = conv2d(max_pool5, weights['wc6'], biases['bc6'])
    print(conv6.shape)
    avg_pool6 = avgPool(conv6, k=2)
    print(avg_pool6.shape)

    pool_flat = tf.reshape(avg_pool6, [-1, weights['w_fc1'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(pool_flat, weights['w_fc1']) + biases['b_fc1'])
    fc1_drop = tf.nn.dropout(fc1, rate = 1 - keep_prob)

    fc2 = tf.nn.relu(tf.matmul(fc1_drop, weights['w_fc2']) + biases['b_fc2'])

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

pred = conv_net(x, weights, biases, keep_prob)

#评估准确率
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#交叉熵预测分类问题
pred_result = tf.argmax(pred, 1)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
train_step = tf.compat.v1.train.AdamOptimizer(learn_rate).minimize(cost)

tf_init = tf.compat.v1.global_variables_initializer()
saver_init = tf.compat.v1.train.Saver()


# 获取屏幕截图并转换为模型的输入
def get_screen_shot():
    # 使用adb命令截图并获取图片
    os.system('adb shell screencap -p /sdcard/girl.png')
    os.system('adb pull /sdcard/girl.png .')

    save_dir = "./test_set/"
    filepath = "./girl.png"
    millisecond = int(round(time.time() * 1000))
    image = cv2.imread(filepath)
    cv2.imwrite('./examples/' + str(millisecond) + "_.png", image)
    #截取有效区域
    image = image[250:1250, 30:1050]
    image = cv2.resize(image, (500, 500))
    #灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_file = face_recognition.load_image_file(filepath)
    #默认使用HOG算法检测人脸
    face_locations = face_recognition.face_locations(gray)
    if len(face_locations) == 0:
        print("尝试使用cnn模型检测人脸")
        face_locations = face_recognition.face_locations(gray, number_of_times_to_upsample=0, model="cnn")
        print("cnn模型检测到人脸数量: " + str(len(face_locations)))
    if len(face_locations) == 0:
        print("can't detect face: " + filepath)
        resize_img = cv2.resize(image, (200, 200))
        filepath = save_dir + str(millisecond) + ".png"
        cv2.imwrite(filepath, resize_img)
    else:
        top, right, bottom, left = face_locations[0]
        height, width, channels = image.shape
        #放大人脸范围
        _left, _top, _right, _bottom = zoomIn(left, top, right, bottom, width, height)
        face = image[_top:_bottom, _left:_right]
        face = cv2.resize(face, (200, 200))
        filepath = save_dir + str(millisecond) + ".png"
        cv2.imwrite(filepath, face)
    return parseImage(filepath, False)


def swipe(score):
    x_start = 850   #default swipe to left
    if score >= 7:
        x_start = 200   #swipe to right with a beautiful girl
    #adb shell input swipe x1 y1 x2 y2 millisecond
    cmd = 'adb shell input swipe {} 688 520 698 {}'.format(x_start, int(random.uniform(100, 150)))
    print(cmd)
    os.system(cmd)


# 解析图片数据
def parseImage(filepath, need_Y_out=True):
    img = cv2.imread(filepath)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img, dtype='float32')
    x_in = np.reshape(img, [width, height, channels])

    for i in range(len(x_in)):
        for j in range(len(x_in[i])):
            x_in[i][j][0] /= 255

    #one-hot encoding
    y_out = np.array([0] * n_classes)
    if need_Y_out:
        score = filepath.split('_')[-1].split('.')[0]
        if score.isdigit():
            y_out[int(score) - 1] = 1
    return x_in, y_out, filepath


# 开始训练
def start_train(sess, epoch):
    path = './training_set/'
    images = os.listdir(path)
    images.sort()
    print(images)
    print('训练集样本数：', len(images))
    total_page = math.ceil(len(images) / batch_size)
    for e in range(epoch):
        loss_array = []
        for page in range(0, total_page):
            imgs = images[page * batch_size:page * batch_size + batch_size]
            batch_xs = []
            batch_ys = []
            for img in imgs[:]:
                filepath = path + img
                # print(filepath)
                x_in, y_out, filepath = parseImage(filepath)
                # print(x_in, y_out)
                batch_xs.append(x_in)
                batch_ys.append(y_out)
            # ————————————————  使用batch_xs，batch_ys训练  ——————————————————
            y_pred, accu, loss, pred_result1, NULL = sess.run([pred, accuracy, cost, pred_result, train_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.6, learn_rate: Learn_rate})
            #输出进度
            ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(ctime, '\t', str(e) + '/' + str(epoch), '\t', str(page) + '/' + str(total_page), "\t", filepath)
            # 保存loss
            loss_array.append(loss)
            #对比结果
            print('truth:', [np.where(i==1)[0][0] for i in batch_ys])
            print('predi:', [i for i in pred_result1])
            print("loss:", '{0:.10f}'.format(loss))
            print("accu:", '{0:.10f}'.format(accu))
            # —————————————————————————————————————————————————————
        saveLoss('./loss.npz', loss_array)            
        saver_init.save(sess, "./model/mode.mod")
    file_writer = tf.compat.v1.summary.FileWriter('./logs', sess.graph)
    print('训练完成！')

#测试模型
def test_model(sess):
    path = './test_set/'
    images = os.listdir(path)
    images.sort()
    print(images)
    print('测试集样本数：', len(images))
    for img in images[:]:
        filepath = path + img
        print(filepath)
        x_in, y_out, filepath = parseImage(filepath, False)
        # print(x_in, y_out)
        batch_xs = [x_in]
        # 使用batch_xs测试
        pred_result1 = sess.run([pred_result], feed_dict={x: batch_xs, keep_prob: 1})
        print(pred_result1[0])
        image = cv2.imread(filepath)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, "score: " + str(pred_result1[0][0] + 1), (50, 180), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("score_image", image)
        cv2.waitKey(2000)   #wait 2 seconds
        cv2.destroyAllWindows()
    print('测试完成！')

def start_play(sess):
    while True:
        x_in, y_out, filepath = get_screen_shot()
        batch_xs = [x_in]
        # result of prediction
        pred_result1 = sess.run(pred_result, feed_dict={x: batch_xs, keep_prob: 1})
        score = pred_result1[0]
        ctime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(ctime, "\tscore: ", score, "\n")
        swipe(score)
        #save score to image
        os.rename(filepath, '.' + filepath.rsplit('.')[1] + '_' + str(score) + '.png')
        time.sleep(random.randrange(200, 700) / 1000)

def saveLoss(filepath, data):
    if os.path.exists(filepath) == False:
        np.savez(filepath, array=data)
    else:
        result = np.load(filepath)['array'].tolist()
        result = result + data
        np.savez(filepath, array=result)


# 区分是train还是play
# IS_TRAINING = True
IS_TRAINING = False
# with tf.device('/gpu:0'):
with tf.compat.v1.Session() as sess:
    sess.run(tf_init)
    model_path = './model/'
    if len(os.listdir(model_path)) > 0:
        saver_init.restore(sess, model_path + 'mode.mod')
    if IS_TRAINING:
        start_train(sess, EPOCH)
    else:
        # test_model(sess)
        start_play(sess)
        # pass



