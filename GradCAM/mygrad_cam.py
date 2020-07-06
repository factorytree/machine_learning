import tensorflow.keras as keras
import cv2
import numpy as np
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from classification.final_Classification_CIFAR import AlexNet

K.set_learning_phase(1) #set learning phase 1:train 0:test
from keras.datasets import  cifar10
# from vgg16 import cifar10vgg16
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# model = cifar10vgg16(train=False,modelpath="D:\Pycharmproject\Machine_Learning\project2\cifar10vgg16_sgd4_epoch29-0.916.hdf5")
model = AlexNet()
model.load_weights('../classification/final_AlexNet_Classification_CIFAR.h5')

x_test=x_test[:50]
prediction = model.predict(x_test)

idx = 2
print(y_test[idx])
print(prediction[idx])
class_idx = np.argmax(prediction[idx])
print("class_idx",class_idx)

print("class_idx",class_idx)
class_output = model.output[:, class_idx]
layer_index=40
# last_conv_layer = model.get_layer(layer_index)
last_conv_layer = model.layers[layer_index]

grads = K.gradients(class_output,last_conv_layer.output)[0]
# pooled_grads = K.mean(grads,axis=(0,1,2))
iterate = K.function([model.input],[grads,last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x_test])
print(np.shape(pooled_grads_value))
print(np.shape(conv_layer_output_value))
# for i in range(512):
#     conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)

img = cv2.resize(x_test[idx], dsize=(32, 32),interpolation=cv2.INTER_NEAREST)
# img = img_to_array(image)
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
img = img.astype(float)
heatmap = heatmap.astype(float)
superimposed_img = cv2.addWeighted(img,0.5,heatmap,0.5,0)
plt_img = superimposed_img[:,:,[2,1,0]]
plt_img /= 255
plt.subplot(2, 1, 1)
plt.imshow(plt_img)
plt.subplot(2, 1, 2)
plt.imshow(x_test[idx]/255)
#plt.savefig(str(idx)+"_"+str(layer_index)+".png")
plt.show()






