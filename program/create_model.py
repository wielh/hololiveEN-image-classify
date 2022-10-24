import keras 
import common_parameter

from keras.layers import Dense,BatchNormalization,Dropout
from keras.models import Model
from keras.applications.efficientnet import EfficientNetB0


# config and package version check
print("keras version: ",keras.__version__)

# create a model
base_model = EfficientNetB0(
    weights='imagenet',
    drop_connect_rate=0.3,
    include_top=False,
    pooling='avg', 
    classes=11, 
    classifier_activation='softmax',
    input_shape=(224,224,3)
)
x =  BatchNormalization(name="final_batch")(base_model.output)
x =  Dropout(rate=0.3, name="final_dropout")(x)
pred = Dense(units=11,activation='softmax', name='output_layer')(x)
model = Model(inputs=base_model.input, outputs=pred)
model.compile(
    optimizer = common_parameter.optimizer,
    loss=common_parameter.loss,
    metrics=['accuracy']
)

model.summary()
model.fit(
    common_parameter.get_train_generator(),
    steps_per_epoch=500,
    epochs=20,
    shuffle=True,
    validation_data=common_parameter.get_test_generator(),
    callbacks=common_parameter.callback
)
model.save('my_model.h5')
#common_parameter.plot_hist(hist)