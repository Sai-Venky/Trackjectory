from pprint import pprint

class Config:

    gpus = '-1'
    num_workers = 0
    print_freq = 1
    save_dir = '/Users/ecom-v.ramesh/Documents/Personal/2020/DL/Trackjectory/output'

    # Multi Tracker
    images_dataset = '/Users/ecom-v.ramesh/Documents/Personal/2020/DL/SOWp2/BDL_FairMOT/tanks/'
    labels_dataset = '/Users/ecom-v.ramesh/Documents/Personal/2020/DL/SOWp2/BDL_FairMOT/tanks/'
    test_images_dataset = '/Users/ecom-v.ramesh/Documents/Personal/2020/DL/Trackjectory/data/kabadi'
    load_model = '/Users/ecom-v.ramesh/Desktop/fairmot_dla34.pth'
    lr = 1.25e-4
    lr_step = [90, 120]
    num_epochs = 3
    batch_size = 1
    save_every = 2
    head_conv = 256
    K = 8
    down_ratio = 4
    wh_weight = 0.1
    off_weight = 1
    hm_weight = 1
    id_weight = 1
    conf_thres = 0.2
    
    # Single Tracker
    single_track = '/Users/ecom-v.ramesh/Documents/Personal/2020/DL/Trackjectory/data/vot2018.txt'
    single_track_batch_size = 10
    single_track_load_model = '/Users/ecom-v.ramesh/Desktop/SiamRPNPPRes50.pth'
    single_track_start_epoch = 0
    single_track_num_epochs = 13
    single_track_lr_warm_epoch = 10
    backbone_train_epoch = 150
    original_lr = 1e-3
    momentum = 0.9
    decay = 0.0001 

    # Tracjectory
    n_stgcnn = 2
    n_txpcnn = 5
    kernel_size = 3
    obs_seq_len = 13
    pred_seq_len = 15
    trajectory_dataset = '/Users/ecom-v.ramesh/Documents/Personal/2020/DL/Trackjectory/data/trajectory'
    trajectory_batch_size = 128
    trajectory_num_epochs = 1000
    trajectory_lr = 0.01
    lr_sh_rate = 150

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        opt.input_h = input_h
        opt.input_w = input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        opt.heads = {'hm': 1,
                    'wh': 4,
                    'id': 128}
        opt.heads.update({'reg': 2})
        opt.img_size = (640, 480)

        return opt

opt = Config()
