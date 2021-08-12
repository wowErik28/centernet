import cv2

from centernet import Centernet

def predicate(img_path):

    centernet = Centernet()
    img = cv2.imread(img_path)
    img = centernet.detect_image(img)
    cv2.imshow('image', img)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    predicate(r'D:\Project\CV\yolo3_learning/VOCdevkit/VOC2007/JPEGImages/000007.jpg')