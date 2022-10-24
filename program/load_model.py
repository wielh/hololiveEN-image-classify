
import tensorflow as tf
import numpy as np
import onnxruntime,cv2,keras

from tensorflow.python.platform import gfile
from os import listdir
from os.path import isfile, join
from common_parameter import get_test_generator,optimizer,loss
from keras.models import model_from_json,load_model
from keras.backend import manual_variable_initialization

# part1 load json and weight
# state: load normally but result is incorrect
# somebody has the same question on github and it is not solved yet (continune)
'''
manual_variable_initialization(True)
test_root_dir = "C:\\Users\\William\\Desktop\\hololive-ai\\test_pictures"
# keras load model
with open("models\\architecture_of_model.json", "r") as f:
    model_1 = model_from_json(f.read())
model_1.load_weights('models\\weights_of_model.h5')
model_1.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
model_1.summary()

# load the image 
test_generator = get_test_generator(test_root_dir)
# evaluate
result = model_1.evaluate(test_generator)
print("[loss,accuracy] = ", result)

# predict
f = test_generator.filenames
predict = model_1.predict(test_generator,steps = len(f))
filenames = test_generator.filenames
class_dict = {
    "Amelia Watson":0, "Ceres Fauna":1, "Gawr Gura":2, "Hakos Baelz":3,
    "IRys":4 , "Mori Calliope":5, "Nanashi Mumei":6, "Ninomea Ina'nis":7,
    "Ouro Kronii":8, "Sana Tsukumo":9, "Takanashi Kiara":10
}

for i in range(len(predict)):  
    if i<50:
        max_num = np.argmax(predict[i])
        exact_class = filenames[i].split("\\")[0]
        pred_class = list(class_dict.keys())[list(class_dict.values()).index(max_num)]  
        print("image path:",filenames[i])
        print("exact class:",exact_class)
        print("predict class:",pred_class)
        print("all probability array:", predict[i])
        print("========")
    else:
        break
print("model_json load done")
print("=========================================")
'''

# part2 load tf pb file
# state: ok
'''
import tensorflow as tf 
import numpy as np
import cv2

from os import listdir
from os.path import join,isfile

# tensorrt
def pb_predict(sess,input_image_nparray):
    sess.run(tf.compat.v1.initialize_all_variables())
    input_plt = tf.compat.v1.placeholder(tf.float32,shape=[1,224,224,3])
    output = sess.graph.get_tensor_by_name('output_layer/bias/rms/Read/ReadVariableOp:0')
    y_pred = sess.run(output, feed_dict = {input_plt: input_image_nparray})
    return y_pred
    
#==================================================================================
print("==============================================")
print("init:")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print("load model:")  
loaded_model = tf.saved_model.load("models/my_model_v5")
print(list(loaded_model.signatures.keys()))
infer = loaded_model.signatures["serving_default"]
print(infer.structured_outputs)
print("{} trainable nodes: {}, ...".format(
          len(loaded_model .trainable_variables),
          ", ".join([v.name for v in loaded_model.trainable_variables[:5]])))

mypath = "test_pictures_2\\test"
pb_model_path = "models\\my_model_v5\\"
index=0

# ref:https://stackoverflow.com/questions/36883949/in-tensorflow-get-the-names-of-all-the-tensors-in-a-graph
# all_operation = tf.compat.v1.get_default_graph().get_operations()
print("predict:")  
for img_path in [f for f in listdir(mypath) if isfile(join(mypath, f))]:
    if index<3:
        img = cv2.imread(join(mypath, img_path),flags=cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img,axis=0)
        img = img.astype(np.float32)
        print("image path:",join(mypath, img_path))
        print("result:", loaded_model(tf.convert_to_tensor(img), training=False).numpy()[0])
        index+=1
        print("========next picture predict===========")
    else:
        break
'''

# part3 load onnx file
# state: ok
'''
# ref:https://onnxruntime.ai/docs/tutorials/tf-get-started.html
all_input = np.zeros(shape=(50,224,224,3))
all_path = []
mypath = "test_pictures_2\\test"
index=0
for img_path in [f for f in listdir(mypath) if isfile(join(mypath, f))]:
    if index<50:
        img1 = cv2.imread(join(mypath, img_path),flags=cv2.IMREAD_COLOR)
        img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_AREA)
        img1 = np.expand_dims(img1,axis=0)
        all_input[index] = img1
        all_path.append(join(mypath, img_path))
        index+=1
    else:
        break
all_input = all_input.astype(np.float32)

class_dict = {
    "Amelia Watson":0, "Ceres Fauna":1, "Gawr Gura":2, "Hakos Baelz":3,
    "IRys":4 , "Mori Calliope":5, "Nanashi Mumei":6, "Ninomea Ina'nis":7,
    "Ouro Kronii":8, "Sana Tsukumo":9, "Takanashi Kiara":10
}
sess = onnxruntime.InferenceSession(
    "models\\my_model_v4.onnx", providers=['CPUExecutionProvider'])
results_ort = sess.run(["output_layer"], {"input_1": all_input})
all_results = results_ort[0]

for i in range(len(all_results)):
    max_num=np.argmax(results_ort[0][i])
    print("img path:",all_path[i])
    #print("prob array:",results_ort[0][i])
    print("predict class num:",max_num)
    print(
        "predict class:",
        list(class_dict.keys())[list(class_dict.values()).index(max_num)]
    )
    print("==============")

print("load onnx model done")
'''