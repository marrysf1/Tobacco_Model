#coding=gbk
import sys
import pandas as pd
import numpy as np
import os
import json
import csv

# 导入openpyxl用于excel操作
from openpyxl import Workbook
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt

from optimization.GeneticAlgorithm import GeneticAlgorithm


def is_nan(nan):
    return nan != nan


acc = 0
Filename = ''


# 读取特征文件  H:/项目开发/烟叶分级/项目程序/Source-Code/MultiModel/Test/log/保山/FeatureMap.txt
def train(dataSet_Path, label, model_name, num_class=3, xgb_param=None):
    header = ['等级', '文件名', '长度', '宽度', '椭圆短长比', '平均宽度', '叶尖夹角', '叶角一阶率', '叶角二阶率',
              '宽度一阶率',
              '宽度二阶率', '单位像素率', '夹角变化差', '烟叶面积', 'RGB-烟叶区域B通道均值', 'RGB-烟叶区域G通道均值',
              'RGB-烟叶区域R通道均值', 'RGB-烟叶区域B通道方差', 'HSV-青率', 'RGB-青率',
              'RGB-烟叶区域G通道方差', '拷红占比', '黑斑占比', '挂灰占比', '橘黄占比', '柠檬黄占比',
              'RGB-烟叶区域R通道方差', 'HSV-烟叶区域H通道均值', 'HSV-烟叶区域S通道均值', 'HSV-烟叶区域V通道均值',
              '聚类色占比', 'HSV-烟叶区域H通道方差', '逆浓度', 'HSV-烟叶区域S通道方差', '能量', '熵', '逆差矩',
              '反均匀度', '对比度', 'HSV-烟叶区域V通道方差',
              'LAB-烟叶区域L通道均值', 'LAB-烟叶区域A通道均值', 'LAB-烟叶区域B通道均值', 'OpenVino青比率',
              'OpenVino杂比率', 'LAB-烟叶区域L通道方差', 'LAB-烟叶区域A通道方差', 'LAB-烟叶区域B通道方差', '重量',
              '保留D']

    # global ks
    # wb = Workbook('result' + '.xlsx')
    # ws = wb.create_sheet('Sheet1')

    # # 打开txt文件，把逗号替换成统一的\t
    # with open(dataSet_Path, 'r') as f:
    #     content = f.read().replace(',', '\t')
    #     lines = content.split('\n')
    #     # 添加表头

    #     ws.append(header)
    #     for line in lines:
    #         item = line.split('\t') 
    #         # item[1] 为文件名，带数字

    #         #保存内容
    #         if(len(item) > 10):
    #             ws.append(item)
    #         #print(item)
    # # 保存excel文件
    # wb.save('result' + '.xlsx')

    # 获取数据源
    # data = pd.read_csv('result' + str(ks) + '.csv',names=header)
    data = pd.read_table(dataSet_Path, sep=',', header=None, names=header)
    data.fillna(0)
    # 读取训练标签
    Y = [label[i] if i in label else i for i in data['等级'].to_list()]  # label

    X = data[header[2:]].values.tolist()  # vector

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123654)

    # print("训练样本数:" + str(len(X_train)) + " " + str(len(y_train)) + "  测试样本数:"  + str(len(X_test)) + " "  + str(len(y_test)))

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': num_class,
        'gamma': xgb_param[0],  # 小数  参数的值越大，算法越保守 0.55
        'max_depth': 24,  # 最大深度 3 - 6 3
        'lambda': xgb_param[2],  # 这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的。 3
        'subsample': xgb_param[3],  # 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合 0.87
        'colsample_bytree': 1,  # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。
        'min_child_weight': int(xgb_param[4]),  # 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。 6
        'eta': xgb_param[5],  # 类似学习率 0.01 - 0.2 0.04
        'seed': 1000,
        'nthread': 4,
    }

    # params = {
    #     'booster': 'gbtree',
    #     'objective': 'multi:softmax',
    #     'num_class': num_class,
    #     'gamma': 0.55, # 小数  参数的值越大，算法越保守 0.55
    #     'max_depth': 3, # 最大深度 3 - 6 3
    #     'lambda': 3.5, # 这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的。 3
    #     'subsample': 0.87,  # 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合 0.87
    #     'colsample_bytree': 1, # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。
    #     'min_child_weight': 6, # 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。 6
    #     'silent': 0,
    #     'eta': 0.04, # 类似学习率 0.01 - 0.2 0.04
    #     'seed': 1000,
    #     'nthread': 4,
    # }
    # print("start training")
    dtrain = xgb.DMatrix(X_train, y_train)

    num_rounds = 500
    model = xgb.train(params, dtrain, num_rounds)

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    ans = model.predict(dtest)

    # 计算准确率
    cnt1 = 0
    cnt2 = 0
    for i in range(len(y_test)):
        if ans[i] == y_test[i]:
            cnt1 += 1
        else:
            cnt2 += 1

    # 从文件读取特征数据并保存至分类源数据

    # 2. 训练模型
    # 3. 保存模型
    curracc = (100 * cnt1 / (cnt1 + cnt2))
    global acc
    global Filename
    if Filename != dataSet_Path:
        Filename = dataSet_Path
        acc = 0
    print("Acc : %.2f %% Accuracy: %.2f %% " % (acc, curracc))
    if curracc > acc:
        acc = curracc
        model.save_model(model_name + '_best.bin')
        print(params)
    model.save_model(model_name + '_last.bin')
    return curracc


def SVMResult(args, vardim, x, bound):
    return train(args[0], args[1], args[2], args[3], x)


model_name_list = ['qj_xg_c', 'qj_xg_x']

if __name__ == "__main__":

    # 遍历特征提取文件夹，解析部位和等级标签
    meragefiledir = 'H:/zhongyanwuliu/DATA/yuezhou/laoshebei/20240831_jianmo/Feature'

    filenames = os.listdir(meragefiledir)
    print(filenames)

    file = open('./Feature/' + 'Feature.txt', 'w', encoding='utf8')

    for filename in filenames:
        filepath = meragefiledir + '\\'
        filepath = filepath + filename
        # 遍历单个文件，读取行数
        for line in open(filepath, encoding='utf8'):
            file.writelines(line)
        # 关闭文件
    file.close()

    file_name_list = ['CFeature.txt', 'XFeature.txt']
    Opfile_list = ['CFeature.txt']

    for i in range(len(Opfile_list)):
        Opfile_list[i] = './Feature/' + Opfile_list[i]

    file_list = []

    for item in Opfile_list:
        file_list.append(open(item, 'w', encoding='utf8'))

    for line in open('./Feature/' + 'Feature.txt', encoding='utf8'):
        for i, item in enumerate(file_name_list):
            if (len(line) < 250):
                continue
            if (item[0] == line[0]):
                file_list[i].writelines(line)
                break

    for i in range(len(file_list)):
        file_list[i].close()

    Level_label = {
        'C2F1': 0,

        'C2F2': 1,
        'C2F3': 2,

        'C3F1': 3,
        'C3F2': 4,
        'C3F3': 5,

        'C3L1': 6,
        'C3L2': 7,
        'C3L3': 8,

        'C4F1': 9,
        'C4F2': 10,
        'C4F3': 11,

    }
    class_number = [12]
    for i, item in enumerate(Opfile_list):
        print(item + model_name_list[i])
        bound = np.array([[0, 2, 2, 0.5, 0.5, 0], [1, 12, 10, 1, 10, 0.2]])
        args = []
        args.append(item)
        args.append(Level_label)
        args.append('./Model/' + model_name_list[i])
        args.append(class_number[i])
        # 初始化遗传算法
        ga = GeneticAlgorithm(50, 6, bound, 5, [0.7, 0.0175, 0.5], SVMResult, args=args)

        # 开始寻优    
        ga.solve()
        # train(item,Level_label,'./Model/' + model_name_list[i],class_number[i])
    print("********************************************")

    # 构建发布模型  --> Release
    # with open('./Release/Demo//Part/thresholdRootPart.json','r') as f:
    #     data = json.load(f)
    #     data['PartClassifyPath'] = './/Config//Part//' + Part_Model_Name
