import math as m
import numpy as np

def minMaxDist(x,y):
    x = np.array(x,np.float16).reshape(-1)
    y = np.array(y,np.float16).reshape(-1)
    map_xy = np.vstack((x,y))
    return np.sum(np.min(map_xy,axis=0))/np.sum(np.max(map_xy,axis=0))

def lsd(a, minx):
    b = []
    c = []
    for i in range(0, len(a)):
        b = a[i]
        for j in range(0, len(b)):
            if j != 2 and j != 1:
                if b[j] <= minx[j]:
                    c.append(1)
                elif b[j] > minx[j]:
                    c.append(round(m.exp(-((b[j] - minx[j]) ** 2)), 4))
            elif j == 2:
                if b[j] <= minx[j]:
                    c.append(0)
                elif b[j] > minx[j]:
                    c.append(round(1 - m.exp(-((b[j] - minx[j]) ** 2)), 4))
            else:
                if b[j] < 0.09:
                    c.append(round(m.exp(-((b[j] - 0.09) ** 2)), 4))
                elif b[j] > 0.09:
                    c.append(round(1 - m.exp(-((b[j] - 0.09) ** 2)), 4))
                else:
                    c.append(1)
    return c


def main():
    s = [[1, 0.09, 37, 0.02],
         [4, 0.36, 12, 0.06],
         [23, 1.8, 2.4, 0.31],
         [110, 0, 0.55, 1.20],
         [660, 0, 0.17, 4.6]]
    t = [[0.559, 0.051, 42, 0.015],
         [22.127, 1.779, 4.942, 0.240],
         [24.736, 2.621, 1.530, 1.182]]
    miny = [1, 0, 0.17, 0.02]
    print(lsd(s, miny))
    print(lsd(t, miny))

    thd1 = []
    for i in range(0, 3):
        p = lsd(t, miny)[0 + i * 4:4 + i * 4]
        for j in range(0, 5):
            q = lsd(s, miny)[0 + j * 4:4 + j * 4]
            thd1.append(minMaxDist(p, q))
    for m in range(0, 3):
        print(thd1[0 + 5 * m:5 + 5 * m])  # 输出每个样本的贴近度


if __name__ == "__main__":
    main()