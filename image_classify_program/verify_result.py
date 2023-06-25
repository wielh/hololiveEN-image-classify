from keras.models import load_model
import numpy as np
from config import IMAGE_WIDTH,IMAGE_HEIGHT, get_validate_generator, get_predict_generator

def inference_validate_picture(model_path:str, val_pictures_path:str, batch_size:int):
    model = load_model(model_path)
    model.summary()
    val_generator = get_validate_generator(val_dir_path = val_pictures_path ,batch_size = batch_size)
    result = model.evaluate(val_generator)
    print("[loss,accuracy] = ", result)

def predict_val_picture(model_path:str):
    model = load_model(model_path)
    model.summary()
    #load the image
    test_generator = get_predict_generator()
    f = test_generator.filenames
    predict = model.predict(test_generator,steps = len(f))
    labels = test_generator.class_indices
    filenames = test_generator.filenames
    print("class type:",labels)

    for i in range(len(predict)):
        exact_class = filenames[i].split("\\")[0]
        max_num = np.argmax(predict[i])
        pred_class = list(labels.keys())[list(labels.values()).index(max_num)]
        if not exact_class == pred_class:
            print("image path:",filenames[i])
            print("exact class:",exact_class)
            print("predict class:", pred_class)
            print("all probability array:", predict[i])
            print("=======================================")

def predict_other_picture(model_path:str , predict_pictures_path:str):

    model = load_model(model_path)
    test_generator = get_predict_generator(sub_dir = "test", test_dir_path = predict_pictures_path)
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
