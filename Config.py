import os


class cfg:
    run_id = "UCSD_Ped2_blur_stride_5"
    mode = "full"  # options: 'full', 'downsamplex2', 'downsamplex4'
    privacy = 'blur'  # options: "raw", "masking", "blur", "background_removal"

    # Paths for UCSD dataset.
    UCSD_train_path = "./UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/"
    UCSD_test_path = "./UCSD/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test"
    
    # Paths for Avenue dataset.
    Avenue_train_path = './Avenue_Dataset/Avenue Dataset'
    Avenue_test_path = './Avenue Dataset/testing_videos/'
    
    # path for pahmdb
    PAHMDB_data_path = './PRIVACY_DATASET/DATASET/hmdb51/'
    PAHMDB_privacy_json_dir = './PA-HMDB51-master/PrivacyAttributes/'
    
    # General paths
    logs = "logs"
    saved_models_dir = "Saved_Models"
    
    # Training parameters
    batch = 4
    epochs = 50
    num_workers = 4
    lr = 1e-4
    num_pa = 5
    stride = 1
    step = 1

    # RELOAD_DATASET = True
    # RELOAD_TESTSET = True
    # RELOAD_MODEL = True
    
    model_path = './model.pth'