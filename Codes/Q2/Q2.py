import numpy as np



def inverse_power_method(A, a):
    print("--Inverse power Iteration Method--\n")
    x = np.random.rand(A.shape[0], 1)
    print("a guess selected for 'a' is: ", a)
    print()
    tempVec = np.eye(len(A))
    tempVec = [a] * tempVec
    tmp = A - tempVec
    eigenVal = 0
    eigVec = []
    y = []
    v = 0
    for _ in range(1000):
        y = np.linalg.solve(tmp, x)
        s = np.linalg.norm(y)
        v = a + 1/s
        x = y/s
        eigVec = x
        eigenVal = v
    return eigenVal, eigVec


A = [[1, 1, 1, 1], [1, 2, 3, 4], [1, 4, 9, 16]]
A = np.array(A)
A = np.transpose(A)
print("A is:\n", A)
b = [[4, 6, 15, 19]]
b = np.array(b)
b = np.transpose(b)
print("\nb is:\n", b)

Z = np.concatenate((A, b), axis=1)
print("\n[A b] is:\n", Z)
U, S, V = np.linalg.svd(Z)

C = Z.T.dot(Z)
print("\nC is:\n", C)
minEigVal, minEigVec = inverse_power_method(C, 1e-10)
print("\nMinimum Eigen Value for [A b] using inverse power method is:\n", minEigVal)
print("\nMinimum Eigen Vector for [A b] using inverse power method is:\n", minEigVec)