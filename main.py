from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('huella2.png')

imageToMatrice = np.asarray(img)
#print(imageToMatrice)
plt.imshow(imageToMatrice)
plt.show()

size=(imageToMatrice.shape[0], imageToMatrice.shape[1],3)
imagen_eg = np.zeros(size)


#ESCALA DE GRISES
for i in range(imagen_eg.shape[0]):
    for j in range(imagen_eg.shape[1]):
        imagen_eg[i, j] = np.mean(imageToMatrice[i, j])/255

plt.imshow(imagen_eg)
plt.show()

#BINARIZADO
for i in range(imagen_eg.shape[0]):
    for j in range(imagen_eg.shape[1]):
        if imagen_eg[i, j, 0] <= 0.55:
            imagen_eg[i, j] = 0
        else:
            imagen_eg[i, j] = 1

#INVERSION A IMAGEN 2D
img_bn_in = np.zeros((imageToMatrice.shape[0], imageToMatrice.shape[1]))
for i in range(imagen_eg.shape[0]):
    for j in range(imagen_eg.shape[1]):
        if imagen_eg[i, j, 0] == 0:
            img_bn_in[i, j] = 1
        else:
            img_bn_in[i, j] = 0



plt.imshow(img_bn_in)
plt.show()
plt.imsave('huellaBN.png', imagen_eg)