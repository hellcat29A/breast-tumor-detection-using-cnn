import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


CATEGORIES = ["negatif", "suspect"]


def prepare(filepath):
    IMG_SIZE = 300  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("32x1-CNN.model")

prediction = model.predict([prepare('test.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
def seuilval(D):
    # equation du seuil
    mu=np.mean(D)
    sigma=np.std(D)
    coef=1.5   #  coef varie de 0 to inf
    Th=(mu+sigma)*np.exp(coef*(np.log(mu/(mu+sigma))))  # selon moi : threshold
    return Th
def affichage(Aim,titre):
    plt.figure()
    plt.imshow(Aim, cmap='bone'); plt.title(titre)
    plt.show()

    plt.figure()
    plt.imshow(Aim, cmap='jet'); plt.title(titre)
    plt.show()

    plt.figure()
    plt.imshow(Aim, cmap='flag'); plt.title(titre)
    plt.show()

A = cv2.imread('test1.jpg',0)
nbit=8
gabor=np.array([[-.38,  1,     1.85,  1,  -.38],
                [1,    -2.61, -4.85, -2.61, 1],
                [1.85, -4.85,  16,   -4.85, 1.85],
                [1,    -2.61, -4.85, -2.61, 1],
                [-.38,  1,     1.85,  1,  -.38]])
FC=cv2.filter2D(A,-1,gabor)
[m,n]=A.shape
seuilA=seuilval(A)
maxA=np.max(A)
maxlevel=2**nbit-1
A0=0*A; AL=0*A; AH=0*A; Aml=0*A
for x in range(0,m):
        for y in range(0,n):
            if A[x,y]==maxA: Aml[x,y]=maxlevel/maxA*A[x,y]
            if A[x,y]!=0 : A0[x,y]=maxlevel
            if (A[x,y]<seuilA) and (A[x,y]!=0) : AL[x,y]=A[x,y]
            if (A[x,y]>=seuilA) and (A[x,y]!=maxlevel) : AH[x,y]=A[x,y]

FCH=cv2.filter2D(AH,-1,gabor)
affichage(A,"A")
affichage(AL,"AL")
affichage(AH,"AH")

prediction = model.predict([prepare('test1.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
def affichage(Aim,titre):
    plt.figure()
    plt.imshow(Aim, cmap='bone'); plt.title(titre)
    plt.show()

    plt.figure()
    plt.imshow(Aim, cmap='jet'); plt.title(titre)
    plt.show()

    plt.figure()
    plt.imshow(Aim, cmap='flag'); plt.title(titre)
    plt.show()

A = cv2.imread('test1.jpg',0)
nbit=8
gabor=np.array([[-.38,  1,     1.85,  1,  -.38],
                [1,    -2.61, -4.85, -2.61, 1],
                [1.85, -4.85,  16,   -4.85, 1.85],
                [1,    -2.61, -4.85, -2.61, 1],
                [-.38,  1,     1.85,  1,  -.38]])
FC=cv2.filter2D(A,-1,gabor)
[m,n]=A.shape
seuilA=seuilval(A)
maxA=np.max(A)
maxlevel=2**nbit-1
A0=0*A; AL=0*A; AH=0*A; Aml=0*A
for x in range(0,m):
        for y in range(0,n):
            if A[x,y]==maxA: Aml[x,y]=maxlevel/maxA*A[x,y]
            if A[x,y]!=0 : A0[x,y]=maxlevel
            if (A[x,y]<seuilA) and (A[x,y]!=0) : AL[x,y]=A[x,y]
            if (A[x,y]>=seuilA) and (A[x,y]!=maxlevel) : AH[x,y]=A[x,y]

FCH=cv2.filter2D(AH,-1,gabor)
affichage(A,"A")
affichage(AL,"AL")
affichage(AH,"AH")

