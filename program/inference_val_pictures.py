from keras.models import load_model
from common_parameter import IMAGE_WIDTH,IMAGE_HEIGHT
from common_parameter import get_test_generator

load_model_name = 'models\\my_model_v5.h5'
test_root_dir = "C:\\Users\\William\\Desktop\\hololive-ai\\test_pictures"

model = load_model(
    load_model_name, 
    #custom_objects={'CustomLayer': CustomLayer}
)
model.summary()

#load the image
test_generator = get_test_generator(test_root_dir)
result = model.evaluate(test_generator)
print("[loss,accuracy] = ", result)
