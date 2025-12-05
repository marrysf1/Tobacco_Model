import numpy as np
import cv2
import csv
import os
import math


#+++++++++++++++图像预处理++++++++++++++#

# 最小外界矩形裁剪
def crop(img):
    img = cv2.copyMakeBorder(img, 500, 500, 500, 500, cv2.BORDER_CONSTANT, value=0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (contours, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    margin = 5  # 裁剪边距
    # draw_rect = image.copy()
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    rect = cv2.minAreaRect(contours[max_idx])  # 得到（中心(x,y); (宽,高); 旋转角度）
    box = np.intp(cv2.boxPoints(rect))  # 获取最小外接矩形的4个顶点坐标
    # cv2.drawContours(img, [box], 0, (255, 255, 255), 2)  # 绘制轮廓最小外接矩形
    h, w = img.shape[:2]
    rect_w, rect_h = int(rect[1][0]) + 1, int(rect[1][1]) + 1
    W = min(rect_w, rect_h)
    H = max(rect_w, rect_h)
    if rect_w <= rect_h:
        x, y = int(box[1][0]), int(box[1][1])  # 旋转中心
        M2 = cv2.getRotationMatrix2D((x, y), rect[2], 1)
        rotated_image = cv2.warpAffine(img, M2, (w * 2, h * 2))
        rotated_canvas = rotated_image[y - margin:y + rect_h + margin + 1, x - margin:x + rect_w + margin + 1]
    else:
        x, y = int(box[2][0]), int(box[2][1])  # 旋转中心
        M2 = cv2.getRotationMatrix2D((x, y), rect[2] + 90, 1)
        rotated_image = cv2.warpAffine(img, M2, (w * 2, h * 2))
        rotated_canvas = rotated_image[y - margin:y + rect_w + margin + 1, x - margin:x + rect_h + margin + 1]
    return W,H, rotated_canvas


#+++++++++++++++灰度图纹理特征++++++++++++++#
# bgr通道均值
def BGR_mean(img):
    B, G, R = cv2.split(img) # B，G，R三（通道）维度分离
    b = B.ravel()[np.flatnonzero(B)]# 将B维度非零数据保留，并展平为一维
    B_mean = sum(b) / len(b)  # sum（）所有像素求和，len（）统计像素点个数
    g = G.ravel()[np.flatnonzero(G)]
    G_mean = sum(g) / len(g)
    r = R.ravel()[np.flatnonzero(R)]
    R_mean = sum(r) / len(r)
    return B_mean, G_mean, R_mean

# hsv通道均值
def HSV_mean(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(imgHSV)
    h = H.ravel()[np.flatnonzero(H)]
    H_mean = sum(h) / len(h)
    s = S.ravel()[np.flatnonzero(S)]
    S_mean = sum(s) / len(s)
    v = V.ravel()[np.flatnonzero(V)]
    V_mean = sum(v) / len(v)
    return H_mean, S_mean, V_mean

# LAB通道均值
def LAB_mean(img):
    imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(imgLAB)
    l = L.ravel()[np.flatnonzero(L)]
    L_mean = sum(l) / len(l)
    a = A.ravel()[np.flatnonzero(A)]
    A_mean = sum(a) / len(a)
    b = B.ravel()[np.flatnonzero(B)]
    B_mean = sum(b) / len(b)
    return L_mean, A_mean, B_mean

# BGR通道方差
def BGR_variance(img):
    b, g, r = cv2.split(img)
    b_var = np.var(b)
    g_var = np.var(g)
    r_var = np.var(r)
    return b_var, g_var, r_var

# hsv通道方差
def HSV_variance(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_var = np.var(h)
    s_var = np.var(s)
    v_var = np.var(v)
    return h_var, s_var, v_var

# lab通道方差
def LAB_variance(img):
    imgLAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(imgLAB)
    l = L.ravel()[np.flatnonzero(L)]
    L_var = np.var(l)
    a = A.ravel()[np.flatnonzero(A)]
    A_var = np.var(a)
    b = B.ravel()[np.flatnonzero(B)]
    B_var = np.var(b)
    return L_var, A_var, B_var


#+++++++++++++++灰度图纹理特征++++++++++++++#
# 定义最大灰度级
gray_level = 16
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    print("图像的高宽分别为：height,width", height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    print("max_gray_level:", max_gray_level)
    return max_gray_level + 1

def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    if d_x >= 0 or d_y >= 0:
        for j in range(height - d_y):
            for i in range(width - d_x):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0
    else:
        for j in range(height):
            for i in range(width):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0
    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)
    return ret

def feature_computer(p):
    mean = 0.0
    Con = 0.0
    Eng = 0.0
    Asm = 0.0
    Idm = 0.0
    Auto_correlation = 0.0
    std2 = 0.0
    std = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            mean += p[i][j] * i / gray_level ** 2
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            Auto_correlation += p[i][j] * i * j
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
        for i in range(gray_level):
            for j in range(gray_level):
                std2 += (p[i][j] * i - mean) ** 2
        std = np.sqrt(std2)
    return mean, Asm, Con, -Eng, Idm, Auto_correlation, std




def read_path(file_pathname):
    # 遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname + '/' + filename)

        background_img = otus_background(img)
        W, H,crop_img = crop(background_img)
        B_mean, G_mean, R_mean = BGR_mean(crop_img)
        H_mean, S_mean, V_mean = HSV_mean(crop_img)
        L_mean, A_mean, B_mean1 = LAB_mean(crop_img)
        b_var, g_var, r_var = BGR_variance(crop_img)
        h_var, s_var, v_var = HSV_variance(crop_img)
        L_var, A_var, B_var = LAB_variance(crop_img)
        img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        glcm_0 = getGlcm(img_gray, 1, 0)
        mean, asm, con, eng, idm, Auto_correlation, std = feature_computer(glcm_0)

       # 数据存储
        header = ['W', 'H',
                         'B_mean', 'G_mean', 'R_mean',
                          'H_mean' , 'S_mean','V_mean',
                          'L_mean','A_mean','B_mean1',
                          'b_var','g_var','r_var',
                          'h_var','s_var','v_var',
                          'L_var','A_var','B_var',
                          'mean','asm','con','eng','idm','Auto_correlation','std']  # 数据列名
        datas = [{ 'W': W, 'H': H,
                        'B_mean': B_mean, 'G_mean': G_mean, 'R_mean': R_mean,
                        'H_mean': H_mean, 'S_mean': S_mean,'V_mean': V_mean,
                        'L_mean': L_mean, 'A_mean': A_mean, 'B_mean1': B_mean1,
                         'b_var': b_var, 'g_var': g_var, 'r_var': r_var,
                         'h_var': h_var, 's_var': s_var, 'v_var': v_var,
                         'L_var': L_var, 'A_var': A_var, 'B_var': B_var,
                         'mean': mean, 'asm': asm, 'con': con,
                         'eng': eng, 'idm': idm, 'Auto_correlation': Auto_correlation,
                         'std': std}]  # 字典数据

        with open('F:/zhengtu/b-red.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
            # writer.writeheader()  # 写入列名
            writer.writerows(datas)  # 写入数据

        cv2.imwrite('F:/zhengtu/1' + "/" + filename, background_img)# 保存图像位置
        cv2.imwrite('F:/zhengtu/2' + "/" + filename, crop_img)  # 保存图像位置

read_path("F:/zhengtu/0")# 输入图像位置
