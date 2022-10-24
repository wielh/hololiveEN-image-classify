from keras.models import load_model
from common_parameter import get_test_generator
import numpy as np

load_model_name = 'models/my_model_v4.h5'
model = load_model(
    load_model_name, 
    #custom_objects={'CustomLayer': CustomLayer}
)
model.summary()

#load the image
test_generator = get_test_generator()
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

