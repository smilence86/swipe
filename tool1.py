import os, cv2
import face_recognition
import matplotlib.pyplot as plt

# path = './examples/'
path = './temp/'
# target_dir = './training_set/'
target_dir = './test_set/'

def prepare_images(dir_from, dir_target):
    images = os.listdir(path)
    for img in images[:]:
        filepath = path + img
        image = cv2.imread(filepath)
        crop_img = image[250:1250, 30:1050]
        resize_img = cv2.resize(crop_img, (500, 500))
        cv2.imwrite(dir_target + img, resize_img)
        # cv2.imshow("cropped", resize_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# prepare_images('./examples/', './training_set/')

# path = './test_set/'
# filepath = path + "Screenshot_20190730-165420.png"
# image = cv2.imread(filepath)
# crop_img = image[250:1250, 30:1050]
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(image, "This one!", (50, 50), font, 2.5, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.imshow("cropped", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#放大人脸范围，增加50像素
def zoomIn(left, top, right, bottom, width, height):
    _left = max(left - 50, 0)
    _top = max(top - 50, 0)
    _right = min(right + 50, width)
    _bottom = min(bottom + 50, height)
    return _left, _top, _right, _bottom

def detect_face(path, target_dir):
    images = os.listdir(path)
    for img in images:
        breaker = False
        for target_img in os.listdir(target_dir):
            if img == target_img:
                breaker = True
                break
        if breaker:
            continue
        filepath = path + img
        print(filepath)
        image = cv2.imread(filepath)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        image = image[250:1250, 30:1050]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h), (0,255,0), 1)
        image = cv2.resize(image, (500, 500))
        cv2.imshow("face_location", image)
        cv2.waitKey(1000)
        # image_file = face_recognition.load_image_file(filepath)
        # face_locations = face_recognition.face_locations(image_file)
        # if len(face_locations) == 0:
        #     print("can't detect face: " + filepath)
        #     # crop_img = image[250:1250, 30:1050]
        #     # resize_img = cv2.resize(crop_img, (200, 200))
        #     # filepath = target_dir + img
        #     # cv2.imwrite(filepath, resize_img)
        # else:
        #     top, right, bottom, left = face_locations[0]
        #     # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        #     height, width, channels = image.shape
        #     # print(image.shape)
        #     #放大人脸范围
        #     _left, _top, _right, _bottom = zoomIn(left, top, right, bottom, width, height)
        #     #画一个矩形
        #     cv2.rectangle(image, (_left, _top), (_right, _bottom), (0,255,0), 1)
        #     #保存脸部特征
        #     # face = image[_top:_bottom, _left:_right]
        #     # face = cv2.resize(face, (200, 200))
        #     # cv2.imwrite(target_dir + img, face)
        #     #显示图片
        #     image = image[250:1250, 30:1050]
        #     image = cv2.resize(image, (500, 500))
        #     cv2.imshow("face_location", image)
        #     cv2.waitKey(1000)
    print("finish")

detect_face('./examples/', './training_set/')