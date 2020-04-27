import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as image


def compressImage(imgLocation, n):
    temp = image.open(imgLocation)
    temp = temp.convert('LA')
    temp = temp.resize((100,100))


    A = np.array(list(temp.getdata(band=0)), float)
    A.shape = (temp.size[1], temp.size[0])
    A = np.matrix(A)
    plt.interactive(False)
    plt.figure(figsize=(9,6))
    plt.title("Original Image resized to 100*100")
    plt.imshow(A, cmap='gray')
    plt.show()
    U, S, V = np.linalg.svd(A)

    x = np.arange(0, len(S), 1)

    plt.scatter(x, S, label="Data", color="green", marker="*", s=30)
    plt.title("Singular Values for %s" %imgLocation)
    plt.show()

    compressedImg = np.matrix(U[:, :n]) * np.diag(S[:n]) * np.matrix(V[:n, :])
    plt.imshow(compressedImg, cmap="gray")
    title = "n = %s" % n
    plt.title(title)
    plt.show()


compressImage("FishScales.JPG", 5)
compressImage("FishScales.JPG", 10)
compressImage("FishScales.JPG", 20)
compressImage("honeycomb.jpg", 5)
compressImage("honeycomb.jpg", 10)
compressImage("honeycomb.jpg", 20)
compressImage("ZebraStripes.jpg", 5)
compressImage("ZebraStripes.jpg", 10)
compressImage("ZebraStripes.jpg", 20)




