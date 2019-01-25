'''
根据log画loss和acc折线图，每一个样本，取其批次求loss和acc
编译ok的：
1.单个打印loss或者acc
2.打印两者到一个坐标中
3.打印两者到一幅图的两个坐标中，正在测试
'''
# from pylab import *
# plt.switch_backend('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

loss_x = []
loss_y = []

# 按批次画图batch_size：64、5
with open("../outputs/vgg_loss.txt") as f:
# with open("./log/loss_acc_22-Jun-06-1529643479_.txt") as f:
    batch_size = 1 #64
    num_f = 0
    len_f = 0
    total_loss = 0.0
    for line in f:
        line = line.strip()
        num_f += 1
        loss_x.append(num_f)
        loss_y.append(float(line.split(' ')[1]))

        # loss_x.append(int(line.split('\t')[0]))
        # loss_y.append(float(line.split('\t')[1]))
        # acc_y.append(float(line.split('\t')[2]))

#不按批次
# with open("./log/loss_acc_25-Jun-06-1529903081_.txt") as f:
#     len_f = 0
#     for line in f:
#         line = line.strip()
#         _loss = float(line.split('\t')[1])
#         _acc = float(line.split('\t')[2])
#         len_f += 1
#         loss_x.append(len_f)
#         acc_x.append(len_f)
#         loss_y.append(_loss)
#         acc_y.append(_acc)
#
#
#         # loss_x.append(int(line.split('\t')[0]))
#         # loss_y.append(float(line.split('\t')[1]))
#         # acc_y.append(float(line.split('\t')[2]))

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(loss_x,loss_y,'b',label="loss")
plt.title('cnn_insurance')
plt.legend(loc='center right')
ax.set_xlabel('step')
plt.ylabel('loss')

# plt.grid(loss_x)#以acc_x为基准显示网格

plt.show()
