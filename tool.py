import os, cv2;
import matplotlib.pyplot as plt;

# path = './examples/'
path = './temp/'
# target_dir = './training_set/'
target_dir = './test_set/'
images = os.listdir(path)

for img in images[:]:
    filepath = path + img
    image = cv2.imread(filepath)
    crop_img = image[250:1250, 30:1050]
    resize_img = cv2.resize(crop_img, (200, 200))
    cv2.imwrite(target_dir + img, resize_img)
    # cv2.imshow("cropped", resize_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# path = './test_set/'
# filepath = path + "Screenshot_20190730-165420.png"
# image = cv2.imread(filepath)
# crop_img = image[250:1250, 30:1050]
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(image, "This one!", (50, 50), font, 2.5, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.imshow("cropped", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()