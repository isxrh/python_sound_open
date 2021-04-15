from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def enframe(x, win, inc=None):
    '''
    分帧函数
    :param x: 语音信号
    :param win: 帧长或窗函数（若为窗函数，帧长则取窗函数长）
    :param inc: 帧移
    :return: 分帧后的(帧数*帧移大小的)二维数组
    '''
    nx = len(x)    # 数据长度
    if isinstance(win, list) or isinstance(win, np.ndarray):
        # 设置了窗函数
        nwin = len(win)
        nlen = nwin  # 帧长=窗长
    elif isinstance(win, int):
        # 未设置窗函数
        nwin = 1
        nlen = win  # 设置为帧长
    if inc is None:
        inc = nlen  # 若未设置帧移，=帧长
    nf = (nx - nlen) // inc + 1  # 计算帧数
    frameout = np.zeros((nf, nlen))     # 初始化
    indf = np.multiply(inc, np.array([i for i in range(nf)]))   # 每帧贼在x中的位移量
    for i in range(nf):
        # 对数据分帧
        frameout[i, :] = x[indf[i]:indf[i] + nlen]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        # 若有窗函数，把每帧乘以窗函数
        frameout = np.multiply(frameout, np.array(win))
    return frameout


if __name__ == "__main__":
    fs, data = wavfile.read('C3_1_y.wav')
    # fs, data, nbits = wavfile.read('C3_1_y.wav')

    inc = 100   # 帧移
    wlen = 200  # 窗长
    en = enframe(data, wlen, inc)
    i = input('起始帧(i):')
    i = int(i)
    tlabel = i
    plt.subplot(4, 1, 1)
    x = [i for i in range((tlabel - 1) * inc, (tlabel - 1) * inc + wlen)]
    plt.plot(x, en[tlabel, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(a)当前波形帧号{}'.format(tlabel))

    plt.subplot(4, 1, 2)
    x = [i for i in range((tlabel + 1 - 1) * inc, (tlabel + 1 - 1) * inc + wlen)]
    plt.plot(x, en[i + 1, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(b)当前波形帧号{}'.format(tlabel + 1))

    plt.subplot(4, 1, 3)
    x = [i for i in range((tlabel + 2 - 1) * inc, (tlabel + 2 - 1) * inc + wlen)]
    plt.plot(x, en[i + 2, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(c)当前波形帧号{}'.format(tlabel + 2))

    plt.subplot(4, 1, 4)
    x = [i for i in range((tlabel + 3 - 1) * inc, (tlabel + 3 - 1) * inc + wlen)]
    plt.plot(x, en[i + 3, :])
    plt.xlim([(i - 1) * inc + 1, (i + 2) * inc + wlen])
    plt.title('(d)当前波形帧号{}'.format(tlabel + 3))

    plt.show()
    plt.savefig('images/en.png')
    plt.close()
