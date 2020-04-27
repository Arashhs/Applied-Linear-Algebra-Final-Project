import numpy as np
import matplotlib.pyplot as plt


def totalLeastSquare(X, Y):
    m, n = X.shape
    Z = np.concatenate((X, Y), axis=1)
    U, S, V = np.linalg.svd(Z)
    V = np.transpose(V)
    VXY = V[0:n, n:]
    VYY = V[n: , n: ]

    B = np.dot(-VXY, np.linalg.pinv(VYY))


    return B



A = [[1, 1, 1, 1], [1, 2, 3, 4], [1, 4, 9, 16]]
A = np.array(A)
A = np.transpose(A)
print("A is:\n", A)
b = [[4, 6, 15, 19]]
b = np.array(b)
b = np.transpose(b)
print("\nb is:\n", b)
tls = totalLeastSquare(A, b)
ls =  np.linalg.solve(A.T.dot(A), A.T.dot(b))
print("\nOrdinary Least Square Answer is\n", ls)
print("\nTotal Least Square Answer is:\n", tls)



x = np.arange(0, 7, 0.1)
y = np.arange(0, 7, 0.1)
Ftls = tls[0] + tls[1]*x + tls[2]*x*x

Fls = ls[0] + ls[1]*x + ls[2]*x*x

x1 = [1, 2, 3, 4]
plt.scatter(x1, b, label= "Data", color= "green", marker= "*", s=30)

plt.plot(x, Ftls, label="Total Least Square", color="blue")
plt.plot(x, Fls, label="Ordinary Least Square", color="red")
plt.legend(loc='upper left')

plt.ylim(-10, 40)
plt.xlim(0,7)


plt.show()

