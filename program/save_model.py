import tensorflow as tf
import onnx,tf2onnx

from keras.models import load_model
from keras import backend as K
# load keras h5 (h5 = GraphDef + CheckPoint)
load_model_name = 'models/my_model_v5.h5'
model = load_model(load_model_name)
model.summary()

# 1.h5 to json + weight 
json_string = model.to_json()
with open('models/architecture_of_model.json', 'w') as file:
    file.write(json_string)
model.save_weights('models/weights_of_model.h5')

# 2. h5 to tf savedmodel(=graphdef(pb file)+checkpoint)
'''
pb stands for protobuf. In TensorFlow, the protbuf file contains 
the graph definition as well as the weights of the model. 
Thus, a pb file is all you need to be able to run a given trained model.
'''
# tf.saved_model.save(model, "models/my_model_v4.pb")
model.save('models/my_model_v5/', save_format="h5")

# 3. tf savedmodel to onnx
'''
ONNX(英語:Open Neural Network Exchange)是一種針對機器學習所設計的開放
式的文件格式，用於存儲訓練好的模型。它使得不同的人工智能框架
(如Pytorch、MXNet)可以採用相同格式存儲模型數據並交互。
'''
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
onnx.save(onnx_model, "models/my_model_v4.onnx")
print(".............")



