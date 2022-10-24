from keras.models import load_model
import numpy as np
from common_parameter import get_predict_generator

load_model_name = 'my_model_v4.h5'
new_pictures_root_dir = "C:\\Users\\William\\Desktop\\hololive-ai\\test_pictures_2"

model = load_model(
    load_model_name, 
    #custom_objects={'CustomLayer': CustomLayer}
)
model.summary()
test_generator = get_predict_generator(
    test_dir_path = new_pictures_root_dir,
    sub_dir = "test"
)

f = test_generator.filenames
predict = model.predict(test_generator,steps = len(f))
filenames = test_generator.filenames

class_dict = {
    "Amelia Watson":0, "Ceres Fauna":1, "Gawr Gura":2, "Hakos Baelz":3,
    "IRys":4 , "Mori Calliope":5, "Nanashi Mumei":6, "Ninomea Ina'nis":7,
    "Ouro Kronii":8, "Sana Tsukumo":9, "Takanashi Kiara":10
}

for i in range(len(predict)):  
    max_num = np.argmax(predict[i])
    print("image path:",filenames[i])
    print("predict class:", list(class_dict.keys())[list(class_dict.values()).index(max_num)])
    print("all probability array:", predict[i])
    print("=======================================")