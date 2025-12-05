import cv2
import numpy as np


def hsv_background(imgbgr):
    imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 90])
    upper = np.array([110, 255, 255])
    mask = cv2.inRange(imghsv, lower, upper)
    ret1, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=2)  # 腐蚀，白区域变小
    mask=cv2.medianBlur(mask,3) # 中值滤波
    mask = cv2.dilate(mask, kernel, iterations=2)  # 膨胀，白区域变大
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = mask.shape[:2]
    mask = np.zeros((h, w, 1), np.uint8)  #
    mask = cv2.drawContours(mask, contours, -1, 1, thickness=-1)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_result = cv2.multiply(imgbgr, mask)
    return img_result


if __name__ == '__main__':
    img0 = cv2.imread('H:/zhongyanwuliu/DATA/yuezhou/xinshebei/20240820/C4F/101302667_-1.png')
    img=hsv_background(img0)


    cv2.namedWindow('th0', 0)
    cv2.imshow("th0", img)
    cv2.namedWindow('th3', 0)
    cv2.imshow("th3", img0)
    cv2.waitKey(0)