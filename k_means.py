import numpy as np
import matplotlib.pylab as plt

num = 100
type = 3
n = num * type
sigma1 = 0.5
sigma2 = 0.5
sigma3 = 0.5
mu1 = 1
mu2 = 2
mu3 = 3
data = np.mat(np.ones((n, 2)))  # 数据矩阵
data[:num, :] = sigma1 * np.random.randn(num, 2) + mu1
data[num:2 * num, :] = sigma2 * np.random.randn(num, 2) + mu2
data[2 * num:3 * num, :] = sigma3 * np.random.randn(num, 2) + mu3

center = np.mat(np.ones((type, 2)))  # 质心矩阵
for i in range(type):
    center[i] = data[i * num]


def dist(vecA, vecB):
    return np.sum((np.power(vecA - vecB, 2)), axis=1)

# 当axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
# 当axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列

J = []
iterNum = 0
# 计算质心并分类
flag = np.mat(np.ones((n, 2)))  # 标记矩阵,第一列是质心索引,第二列是误差大小以便评价性能
# clusterChanged = True
while True:
    for i in range(n):
        minDist = np.iinfo(np.int16).max
        minIndex = -1
        for j in range(type):
            Dist = dist(data[i], center[j])
            if Dist < minDist:
                minDist = Dist
                minIndex = j
        flag[i, :] = minIndex, minDist
    J.append(sum(flag[:, 1])[0, 0])
    iterNum = iterNum + 1
    # 更新质心位置
    for i in range(type):
        # np.where 获得一个符合条件的目录，然后取[0] 代表的是取的某一行，否则取的是具体位置上的数
        temp = data[np.where(flag[:, 0].A == i)[0]]
        center[i, :] = np.mean(temp, axis=0)
    length = len(J);
    if length > 2 and J[length - 1] == J[length - 2]:
        break
print("\n质心坐标:\n", center)
print("\n质心索引和误差大小\n", flag)
print("\n误差和\n", J)
# 画图
plt.figure(1, dpi=100)
plt.figure()
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.scatter(data[:num, 0].tolist(), data[:num, 1].tolist(), c='r', s=30, alpha=0.2, marker='D')
plt.scatter(data[num:2 * num, 0].tolist(), data[num:2 * num, 1].tolist(), c='b', s=30, alpha=0.2, marker='o')
plt.scatter(data[2 * num:3 * num, 0].tolist(), data[2 * num:3 * num, 1].tolist(), c='g', s=30, alpha=0.2, marker='*')
plt.show();

plt.figure(2, dpi=100)
plt.figure()
plt.xlabel("X axis")
plt.ylabel("Y axis")
x1 = [];
y1 = [];
x2 = [];
y2 = [];
x3 = [];
y3 = [];
for i in range(n):
    if flag[i, 0] == 0:
        x1.append(data[i, 0]);
        y1.append(data[i, 1])
    elif flag[i, 0] == 1:
        x2.append(data[i, 0]);
        y2.append(data[i, 1])
    else:
        x3.append(data[i, 0]);
        y3.append(data[i, 1])
# s:大小 alpha:透明度 maeker:形状
plt.scatter(x1, y1, c='r', s=30, alpha=0.2, marker='D')
plt.scatter(center[0, 0], center[0, 1], c='r', s=150, alpha=1, marker='D')
plt.scatter(x2, y2, c='b', s=30, alpha=0.2, marker='o')
plt.scatter(center[1, 0], center[1, 1], c='b', s=200, alpha=1, marker='o')
plt.scatter(x3, y3, c='g', s=30, alpha=0.2, marker='*')
plt.scatter(center[2, 0], center[2, 1], c='g', s=300, alpha=1, marker='*')
# 显示所画的图
plt.show()

plt.figure(3, dpi=100)
x = range(0, iterNum)
# 画出loss下降趋势
y = J
plt.plot(x[int(len(J) / 2):], y[int(len(J) / 2):], color="k", label='loss')
plt.legend()
# 去掉右边框和上边框
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 显示所画的图
plt.show()

# X = np.mat(np.arange(2,6,1).reshape(2,2))
# Y = np.mat(np.arange(4).reshape(2,2))
# print(X)
# print(Y)
# print("sum:",np.sum(X[:,0]))

# print(np.where(Y[:,0].A==2))
# temp = data[np.where(Y[:,0].A==2)[0]]
# print("temp\n",temp)
# print("data\n",data)
