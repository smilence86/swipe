# -*- coding: utf-8 -*-

import random, time as time, datetime, shutil, os, cv2
import numpy as np
from PIL import Image
import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import *

# matplotlib.use('qt4agg')

myfont = FontProperties(fname='./SourceHanSerifCN-Light.otf')


def showGraph(filepath):
    arr = np.load(filepath)['array'].tolist()[:]
    # arr = np.load(filepath)['array'].tolist()[-50:]
    print(len(arr))
    # for t in arr:
    #     if t > 0.3:
            # print('{0:.10f}'.format(t))
    print(arr)
    x = np.arange(len(arr))
    print(len(x))
    y = arr
    # plt.figure()
    matplotlib.rcParams['axes.unicode_minus']=False  
    plt.title(u'loss趋势', fontproperties=myfont)
    plt.xlabel('训练次数', fontproperties=myfont)
    plt.ylabel('loss')
    plt.plot(x, y)
    plt.show()

showGraph('./loss.npz')


