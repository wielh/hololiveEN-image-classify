import argparse
from image_classify_program.model_actions import create_model,retrain_model, model_transform
from image_classify_program.verify_result import inference_validate_picture, predict_other_picture
from image_classify_program.grad_cam import show_result

def start(type_of_actions:list):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', help='要執行的動作, 選項:'+(', '.join(type_of_actions)))
    parser.add_argument('--train_root_dir')
    parser.add_argument('--val_root_dir')
    parser.add_argument('--test_root_dir')

    parser.add_argument('--epochs')
    parser.add_argument('--step_per_epoch')
    parser.add_argument('--batch_size')

    parser.add_argument('--model_path')
    parser.add_argument('--load_model_path')
    parser.add_argument('--save_model_path')
    parser.add_argument('--save_architecture_path')
    parser.add_argument('--save_weight_path')

    parser.add_argument('--learing_rate')
    parser.add_argument('--last_conv_layer_name')
    parser.add_argument('--model_type')
    args = parser.parse_args()

    if not args.action:
        print('Pleas input actions')
        return
    elif args.action not in  type_of_actions:
        print('Action has to be one of the following:' +(', '.join(type_of_actions)))
        return

    '''
    actions
    1. create model, args=batch_size,epoch,step_per_epoch,model_path
    2. retrain model
    3. model inference
    4. model predict
    5. model type transform
    6. grad_cam to verify model
    '''
    try:
        if args.action == 'create_model':
            create_model(
                train_root_dir = args.train_root_dir,
                val_root_dir = args.val_root_dir,
                learing_rate = args.learing_rate,
                epochs = int(args.epochs),
                step_per_epoch = int(args.step_per_epoch),
                batch_size= int(args.batch_size),
                model_path= args.model_path
            )
        elif args.action == 'retrain_model':
            retrain_model(
                train_root_dir = args.train_root_dir,
                val_root_dir = args.val_root_dir,
                learing_rate = args.learing_rate,
                epochs = int(args.epochs),
                step_per_epoch = int(args.step_per_epoch),
                batch_size= int(args.batch_size),
                load_model_path = args.load_model_path,
                save_model_path= args.save_model_path
            )
        elif args.action  == 'model_inference':
            inference_validate_picture(
                model_path = args.load_model_path,
                val_pictures_path = args.val_root_dir,
                batch_size = int(args.batch_size)
            )
        elif args.action  == 'model_predict':
            predict_other_picture(
                model_path = args.load_model_path,
                predict_pictures_path = args.test_root_dir
            )
        elif args.action == 'model_type_transform':
            model_transform(args)
        elif args.action  == 'grad_cam':
            show_result(
                load_model_name= args.load_model_path,
                last_conv_layer_name = args.last_conv_layer_name,
                test_dir_path = args.test_root_dir
            )
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        print("Execution complete.")

type_of_actions = ['create_model','retrain_model','model_inference',
    'model_predict', 'model_type_transform', 'grad_cam']
start(type_of_actions)
