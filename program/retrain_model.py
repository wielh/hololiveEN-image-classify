
import common_parameter
from keras.models import load_model

# config and package version check
load_model_name = 'models/my_model_v5/'
save_model_name = 'models/my_model_v5.h5'
batch_size=8

model = load_model(load_model_name)
model.compile(
    optimizer = common_parameter.optimizer,
    loss=common_parameter.loss,
    metrics=['accuracy']
)
model.summary()
hist = model.fit(
    common_parameter.get_train_generator(),
    steps_per_epoch=500,
    epochs=20,
    shuffle=True,
    validation_data=common_parameter.get_test_generator(),
    callbacks=common_parameter.callback
)
model.save(save_model_name)
common_parameter.plot_hist(hist)