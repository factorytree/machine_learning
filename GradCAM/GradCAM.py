import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.utils import np_utils
import cv2
from classification.final_Classification_CIFAR import AlexNet


def grad_cam(model, x, x_ori, category_index, idx):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """

    # 取得目标分类的CNN输出值
    class_output = model.output[:, category_index]
    print(class_output)

    # 取得想要算出梯度的层的输出
    # convolution_output = model.layers[layer_number].output
    convolution_output = model.get_layer('batch_normalization_v1_16').output
    print(convolution_output)


    # 利用gradients函数，算出梯度公式
    grads = K.gradients(class_output, convolution_output)[0]
    # 定义计算函数（tensorflow的常见做法，与一般开发语言不同，先定义计算逻辑图，之后一起计算。）
    gradient_function = K.function([model.input], [convolution_output, grads])
    # 根据实际的输入图像得出梯度张量(返回是一个tensor张量)
    output, grads_val = gradient_function(x)
    output, grads_val = output[idx], grads_val[idx]

    # 取得所有梯度的平均值(维度降低：2X2X384 -> 384)
    weights = np.mean(grads_val, axis=(0, 1))
    # 把所有平面的平均梯度乘到最后卷积层上，得到一个影响输出的梯度权重图
    cam = np.dot(output, weights)

    # 把梯度权重图RGB化
    cam = cv2.resize(cam, (x.shape[1], x.shape[2]))
    cam = np.maximum(cam, 0)
    heatmap = (cam-np.min(cam)) / (np.max(cam)-np.min(cam))

    # Return to BGR [0..255] from the preprocessed image
    image_rgb = x_ori[idx, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)

    cam = cv2.applyColorMap(np.uint8(255 * (1-heatmap)), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    # print(np.shape(cam))
    return np.uint8(cam), heatmap, np.uint8(image_rgb)

model = AlexNet()
model.load_weights('../classification/final_AlexNet_Classification_CIFAR.h5')


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test1 = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test1 / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

idx = 35
x_test=x_test[:100]
prediction = model.predict(x_test)


print(y_test[idx])
print(prediction[idx])
class_idx = np.argmax(prediction[idx])
print("class_idx",class_idx)

img_heatmap, heatmap, img=grad_cam(model, x_test, x_test1, class_idx, idx)
x=cv2.resize(x_test[0], (32,32), cv2.INTER_LINEAR)
plt.subplot(3, 1, 1)
plt.imshow(img)
plt.subplot(3, 1, 2)
plt.imshow(heatmap)
plt.subplot(3,1,3)
plt.imshow(img_heatmap)
plt.savefig("heatmap/"+str(idx)+".png")
plt.show()