import numpy as np
import matplotlib.pylab as plt

num1 = 20
num2 =50
num3 = 100
K = 3  # type
n = num1 + num2 + num3
sigma1 = 0.2
sigma2 = 0.3
sigma3 = 0.3
mu1 = 1
mu2 = 2
mu3 = 3
data = np.mat(np.ones((n, 2)))  # 数据矩阵
data[:num1, :] = sigma1 * np.random.randn(num1, 2) + mu1
data[num1:num1 + num2, :] = sigma2 * np.random.randn(num2, 2) + mu2
data[num1 + num2:n, :] = sigma3 * np.random.randn(num3, 2) + mu3

R = np.mat(np.ones((n, K)))  # Rjk表示j属于k类的概率
R[:,0] = num1/n
R[:,1] = num2/n
R[:,2] = num3/n
U = np.mat([[1.0,1.0],[2.0,2.0],[3.0,3.0]])  # 均值
A = np.mat(np.ones((K, 1)))  # 各类占总的比例
A[0] = num1/n
A[1] = num2/n
A[2] = num3/n
E = []  # 协方差
# for k in range(K):
#     Nk = np.sum(R[:,0])
#     sumE = np.mat(np.zeros((2, 2)))
#     for j in range(n):
#         sumE = sumE + R[j, k] * (data[j] - U[k]).T * (data[j] - U[k])
#     E.append(sumE / Nk)
for i in range(K):
    E.append(np.mat(np.eye(2, dtype=int)))



def gaussian(X, U, E):  # 多维高斯密度函数
    e = np.linalg.det(E)
    if e==0:
        print("!!!=0")
        exit(-1)
    a = 1/(2*np.pi)/np.power(e,0.5)
    b = (X-U)*E.I*(X-U).T
    return a*np.exp(-0.5*b)

# EM
iterNum = 100
for i in range(iterNum):
    # E:
    for j in range(n):
        sumRij = 0
        for k in range(K):
            gau2 = gaussian(data[j], U[k], E[k])
            sumRij = sumRij + A[k] * gau2
        for k in range(K):
            gau1 = gaussian(data[j], U[k], E[k])
            R[j, k] = A[k] * gau1 / sumRij
    # M:
    for k in range(K):
        Nk = np.sum(R[:,k])

        U[k] = R[:, k].T * data / (1.0*Nk)
        A[k] = Nk / (n*1.0)
        sumE = np.mat(np.zeros((2, 2)))
        for j in range(n):
            sumE = sumE + R[j, k] * (data[j] - U[k]).T * (data[j] - U[k])
        E[k] = sumE / (1.0*Nk)
        # print(U[k])

print("U:\n", U)
print("A:\n", A)
print("E:\n", E)

# data
plt.figure(1, dpi=100)
plt.figure()
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.scatter(data[:num1, 0].tolist(), data[:num1, 1].tolist(), c='r', s=30, alpha=0.2, marker='D')
plt.scatter(data[num1:num1 + num2, 0].tolist(), data[num1:num1 + num2, 1].tolist(), c='b', s=30, alpha=0.2, marker='o')
plt.scatter(data[num1 + num2:n, 0].tolist(), data[num1 + num2:n, 1].tolist(), c='g', s=30, alpha=0.2, marker='*')

plt.scatter(U[:, 0].tolist(), U[:, 1].tolist(), c='y', s=200, alpha=0.6, marker='o')
plt.show()

# X = np.mat(np.arange(2,6,1).reshape(2,2))
# Y = np.mat(np.arange(4).reshape(2,2))
# print(X)
# print(Y)
# print("sum:",sum(X[:,0]))
# print(np.linalg.det(Y))
