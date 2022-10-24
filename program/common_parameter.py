from keras.preprocessing.image import ImageDataGenerator
from keras.optimizer_v2.adam import Adam
from keras.optimizer_v2.rmsprop import RMSProp
from keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN

import matplotlib.pyplot as plt
import tensorflow as tf

IMAGE_WIDTH=224
IMAGE_HEIGHT=224
batch_size=6
train_root_dir='C:\\Users\\William\\Desktop\\hololive-ai\\train_pictures'
test_root_dir='C:\\Users\\William\\Desktop\\hololive-ai\\test_pictures'
# https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

# reference: https://stackoverflow.com/questions/69746393/keras-discrepancy-between-evaluate-and-predict
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    rotation_range=0, 
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    channel_shift_range=8,
    brightness_range=[0.9,1.1],
    fill_mode='constant',cval=0,
    horizontal_flip=True,
    vertical_flip=True,
)

test_datagen = ImageDataGenerator(
    # rescale=1./255,
    fill_mode='constant',cval=0,
    horizontal_flip=False,
    vertical_flip=False,
)

callback = [
    ModelCheckpoint(
        filepath="models/checkpoint",
        monitor="val_loss",
        verbose=0,
        save_weights_only=True
    ),
    EarlyStopping(
        monitor="val_loss", min_delta=0, patience=5 ,verbose=0
    ),
    TerminateOnNaN()
]

def get_train_generator(train_dir_path: str = train_root_dir):
    return train_datagen.flow_from_directory(
        train_dir_path,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode = 'rgb',
        shuffle=True,seed=23459
    )

def get_test_generator(
    test_dir_path: str = test_root_dir,
    batch_size: int= batch_size):
    return test_datagen.flow_from_directory(
        test_dir_path, 
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=batch_size,
        class_mode = 'categorical',
        color_mode = 'rgb',
        shuffle=False,
    )

def get_predict_generator(sub_dir:str, test_dir_path:str = test_root_dir):
    return test_datagen.flow_from_directory(
        test_dir_path,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=1,
        classes=[sub_dir],
        class_mode=None,
        color_mode='rgb',
        shuffle=False     
    )

optimizer = RMSProp(
    learning_rate=ExponentialDecay(
        initial_learning_rate=1e-7,
        decay_steps=1000,
        decay_rate=0.95,
        staircase=True
    ),
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
)

'''
optimizer=Adam(
    learning_rate=ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=50,
        decay_rate=0.9,
        staircase=True
    )
)
'''

loss = CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
)

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

if __name__=="__main__":
    for _ in range(5):
        img, label = get_train_generator().next()
        print(label)
        print(img.shape)   #  (1,256,256,3)
        plt.imshow(img[0])
        plt.show()
        