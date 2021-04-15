import numpy as np
from enframe import enframe


def STAc(x):
    """
    计算短时相关函数
    :param x:
    :return:
    """
    para = np.zeros(x.shape)
    fn = x.shape[1]
    for i in range(fn):
        R = np.correlate(x[:, i], x[:, i], 'valid')
        para[:, i] = R
    return para


def STEn(x, win, inc):
    """
    计算短时能量函数
    :param x: 语音信号
    :param win: 窗函数或帧长
    :param inc: 帧移
    :return: 短时能量
    """
    X = enframe(x, win, inc)
    s = np.multiply(X, X)
    return np.sum(s, axis=1)


def STMn(x, win, inc):
    """
    计算短时平均幅度计算函数
    :param x: 语音信号
    :param win: 窗函数或帧长
    :param inc: 帧移
    :return: 短时平均幅度
    """
    X = enframe(x, win, inc)
    s = np.abs(X)
    return np.mean(s, axis=1)


def STZcr(x, win, inc, delta=0):
    """
    计算短时过零率
    :param x: 语音信号
    :param win: 窗函数或帧长
    :param inc: 帧移
    :return: 短时过零率
    """
    X = enframe(x, win, inc)    # 分帧
    fn = X.shape[0]             # 求出帧数
    wlen = X.shape[1]           # 求出帧长
    para = np.zero(1,fn)        # 初始化
    for i in range(fn):
        z = X[i,:]      # 取得一帧数据
        for j in range(wlen):   # 在一帧内寻找过零点
            if(z[j]*z[j+1] < 0):    # 判断是否为过零点
                para[i] = para[i] + 1;  # 过零点，记录1次
    return para

    # absx = np.abs(x)
    # x = np.where(absx < delta, 0, x)
    # X = enframe(x, win, inc)
    # X1 = X[:, :-1]
    # X2 = X[:, 1:]
    # s = np.multiply(X1, X2)
    # sgn = np.where(s < 0, 1, 0)
    # return np.sum(sgn, axis=1)


def STAmdf(X):
    """
    计算短时幅度差，好像有点问题
    :param X:
    :return:
    """
    # para = np.zeros(X.shape)
    fn = X.shape[1]
    wlen = X.shape[0]
    para = np.zeros((wlen, wlen))
    for i in range(fn):
        u = X[:, i]
        for k in range(wlen):
            en = len(u)
            para[k, :] = np.sum(np.abs(u[k:] - u[:en - k]))
    return para


def FrameTimeC(frameNum, frameLen, inc, fs):
    ll = np.array([i for i in range(frameNum)])
    return ((ll - 1) * inc + frameLen / 2) / fs
