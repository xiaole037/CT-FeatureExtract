'''
根据log画acc折线图
'''
# from pylab import *
# plt.switch_backend('agg')
import matplotlib.pyplot as plt

l_n = 0
a_n = 0
loss_x = []
loss_y = []
acc_x = []
acc_y = []
with open("../outputs/vgg_loss.txt") as f:
    for line in f:
        line = line.strip()
        acc_y_value = float(line.split(' ')[1])
        if acc_y_value > 10.00:
            continue
        acc_x.append(int(line.split(' ')[0]))

        acc_y.append(acc_y_value)
        # if len(line.split("accuracy = ")) == 2:
        #     acc_y.append(float(line.split("accuracy = ")[1]))
        #     acc_x.append(a_n)
        #     a_n += 100   #根据test_interval调整
        # if len(line.split(" loss = ")) == 2:
        #     loss_y.append(float(line.split(" loss = ")[1]))
        #     loss_x.append(l_n)
        #     l_n += 20   #根据display调整
# print(len(loss_x))
plt.figure(figsize=(8,6))
# plt.plot(loss_x,loss_y,'',label="loss")
plt.plot(acc_x,acc_y,'',label="loss")
plt.title('loss')
plt.legend(loc='upper right')
plt.xlabel('iter')
plt.ylabel('')
# plt.grid(loss_x)
plt.show()
# plt.savefig('../outputs/vgg_loss.png')
