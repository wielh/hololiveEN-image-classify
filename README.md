# hololiveEN-image-classify

本程式演示了如何用 EfficientNet 快速的構建一個可以進行影像分類的 AI

1. 簡介: 程式有以下的 action 可以使用: 生成模型、載入模型、預測和 grad_cam (驗證model的合理性)

2. 生成模型: 使用以下指令

    python app.py --action create_model --train_root_dir <str> --val_root_dir <str> --learing_rate <float> --epochs <int> --step_per_epoch <int> --batch_size <int> --model_path <str>

3. 載入模型繼續訓練: 使用以下指令

    python app.py --action retrain_model --train_root_dir <str> --val_root_dir <str> --learing_rate <float> --epochs <int> --step_per_epoch <int> --batch_size <int> --load_model_path <str> --save_model_path <str>

4. 預測結果: 使用以下指令

    python app.py --action model_predict --model_path <str> --predict_pictures_path <str>

5. grad_cam

  grad_cam 是一種方法，用來顯示 AI 是看突變的哪個地方作為分類依據。指令是:

    python app.py --action model_predict --grad_cam <str> --last_conv_layer_name <str> --test_dir_path <str>

  其中 last_conv_layer_name 可以從 model_summary 中取得。他是模型的最後一個convolution layer的名字





