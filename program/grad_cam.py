import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os, cv2

from common_parameter import get_test_generator,get_predict_generator
from keras.models import load_model,Model

def gradcam(model, input_image, last_convolution_layer_name:str):
    # 取得影像的分類類別
    preds = model.predict(input_image)
    pred_class = np.argmax(preds[0])
    print("predict class index:",pred_class)
    # 預測分類的輸出向量
    # pred_output = model.output[:, pred_class]
    heatmap_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 求得分類的神經元對於最後一層 convolution layer 的梯度
    with tf.GradientTape() as tape:
      conv_output, predictions = heatmap_model(input_image)
      most_possible_val = predictions[:, np.argmax(predictions[0])]
    # grad(dy/dx) (x=x_0,y=y_0) =  tape.gradient(y_0,x_0)
    grads = tape.gradient(most_possible_val, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    '''
    deprecated:
      grads = K.gradients(pred_output, last_conv_layer.output)[0]  
      pooled_grads = K.sum(grads, axis=(0, 1, 2))
    '''
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def plot_heatmap(heatmap, img_path, pred_class_name):

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) 
    original_img = cv2.imread(img_path)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    img = cv2.resize(
        cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), 
        (original_img.shape[1], original_img.shape[0])
    )

    plt.imshow(img)
    plt.axis('off')
    plt.title("original")

    fig.add_subplot(1, 2, 2)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap) 
    plt.imshow(img, alpha=0.5)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)   
    plt.axis('off')
    plt.title("heatmap,predict class:"+ pred_class_name)
    plt.show()

load_model_name = 'my_model_v4.h5'
last_conv_layer_name = 'top_conv'
model = load_model(load_model_name)
model.layers[-1].activation = None   #???
model.summary()

root_dir="C:\\Users\\William\\Desktop\\hololive-ai\\test_pictures"
# test_generator = get_predict_generator(sub_dir="Amelia Watson",test_dir_path=root_dir)
test_generator = get_test_generator(test_dir_path=root_dir,batch_size=1)
filenames = test_generator.filenames
predict = model.predict(test_generator,steps = len(filenames))
labels = test_generator.class_indices
print("======================================")

for i in range(len(predict)):  
    max_num = np.argmax(predict[i])
    pred_class = list(labels.keys())[list(labels.values()).index(max_num)]
    filename = os.path.join(root_dir,filenames[i])
    print("image path:",filename)
    print("image exist:",os.path.exists(filename))
    print("predict class:", pred_class)
    print("all probability array:", predict[i])
    print("=======================================")
    heatmap = gradcam(model, test_generator[i][0],last_conv_layer_name)
    plot_heatmap(heatmap, filename, pred_class)