import config
import onnx,tf2onnx

from keras.models import load_model
from keras.layers import Dense,BatchNormalization,Dropout
from keras.models import Model
from keras.applications.efficientnet import EfficientNetB0

def create_model(train_root_dir:str, val_root_dir:str, learing_rate:str,
    epochs:int, step_per_epoch:int , batch_size:int, model_path:str):
    # config and package version check
    # print("keras version: ",keras.__version__)
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
        optimizer = config.get_optimizer(learing_rate),
        loss=config.loss,
        metrics=['accuracy']
    )

    model.summary()
    model.fit(
        config.get_train_generator(train_dir_path=train_root_dir,batch_size=batch_size),
        steps_per_epoch=step_per_epoch,
        epochs=epochs,
        shuffle=True,
        validation_data=config.get_validate_generator(val_dir_path=val_root_dir,batch_size=batch_size),
        callbacks=config.callback
    )
    model.save(model_path)

def retrain_model(train_root_dir:str, val_root_dir:str, learing_rate:str,
    epochs:int, step_per_epoch:int , batch_size:int,
    load_model_path:str, save_model_path:str):

    model = load_model(load_model_path)
    model.compile(
        optimizer = config.get_optimizer(learing_rate),
        loss=config.loss,
        metrics=['accuracy']
    )
    model.summary()
    model.fit(
        config.get_train_generator(train_dir_path=train_root_dir,batch_size=batch_size),
        steps_per_epoch=step_per_epoch,
        epochs=epochs,
        shuffle=True,
        validation_data=config.get_validate_generator(val_dir_path=val_root_dir,batch_size=batch_size),
        callbacks=config.callback
    )
    model.save(save_model_path)

#=====================================================================

type_of_model = ['h5','savedmodel','onnx']

def model_transform(args):
    if not args.model_type:
        print('Pleas input model_type')
        return
    elif args.model_type not in type_of_model:
        print('Action has to be one of the following:' +(', '.join(type_of_model)))
        return

    if args.model_type == 'h5':
        keras_to_h5(
            load_model_path = args.load_model_path,
            save_architecture_path = args.save_architecture_path,
            save_weight_path = args.save_weight_path
        )
    elif args.model_type == 'savedmodel':
        keras_to_savedmodel(
            load_model_path= args.load_model_path,
            save_model_path= args.save_model_path
        )
    elif args.model_type  == 'onnx':
        keras_to_onnx(
            load_model_path = args.load_model_path,
            save_model_path= args.save_model_path
        )
    return

# keras to h5 (h5 = GraphDef + CheckPoint)
def keras_to_h5(load_model_path:str, save_architecture_path:str, save_weight_path:str):
    # load keras h5 (h5 = GraphDef + CheckPoint)
    model = load_model(load_model_path)
    json_string = model.to_json()
    with open(save_architecture_path, 'w') as file:
        file.write(json_string)
    model.save_weights(save_weight_path)

#  h5 to tf savedmodel(=graphdef(pb file)+checkpoint)
'''
pb stands for protobuf. In TensorFlow, the protbuf file contains
the graph definition as well as the weights of the model.
Thus, a pb file is all you need to be able to run a given trained model.
'''
def keras_to_savedmodel(load_model_path:str, save_model_path:str):
    model = load_model(load_model_path)
    model.save(save_model_path, save_format="h5")

#  tf savedmodel to onnx
'''
ONNX(英語:Open Neural Network Exchange)是一種針對機器學習所設計的開放
式的文件格式，用於存儲訓練好的模型。它使得不同的人工智能框架
(如Pytorch、MXNet)可以採用相同格式存儲模型數據並交互。
'''
def keras_to_onnx(load_model_path:str, save_model_path:str):
    model = load_model(load_model_path)
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
    onnx.save(onnx_model, save_model_path)
