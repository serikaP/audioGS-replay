from yacs.config import CfgNode as CN

cfg = CN()

cfg.device = 'cuda'

cfg.dist_backend = 'nccl'

cfg.log_dir = 'logs/'
cfg.output_dir = 'outputs/'
cfg.result_dir = 'results/'

cfg.seed = 42

cfg.workers = 4

cfg.pi = 'psnr'

cfg.model = ''


# dataset
cfg.dataset = CN()

cfg.dataset.img_num_per_gpu = 1

cfg.dataset.N_points = 256
cfg.dataset.H = 1024
cfg.dataset.W = 1024
cfg.dataset.name = ''
cfg.dataset.video = 1
cfg.dataset.data_root = 'data/'
cfg.dataset.visible = False
cfg.dataset.sr = 44100
cfg.dataset.audio_len = 2.0
cfg.dataset.scene_normalize = True
cfg.dataset.scene_scope = ''
cfg.dataset.test_viewpoint = 0
cfg.dataset.viewpoint_split = False
cfg.dataset.selected_scenes = []
# Camera pose source for ReplayNVAS Audio3DGS loaders:
# - 'fixed_rotation': use camera_positions_fixed_rotation.json under data_root
# - 'gs_cameras': use cam_imags/gs_cameras.json (COLMAP/SfM coordinate system)
# - 'replay_metadata': use original Replay metadata.sqlite camera extrinsics
cfg.dataset.pose_source = 'fixed_rotation'
cfg.dataset.fixed_pose_file = 'camera_positions_fixed_rotation.json'
cfg.dataset.gs_cameras_file = 'cam_imags/gs_cameras.json'
cfg.dataset.replay_metadata_file = 'data/Replay/metadata.sqlite'
# How to map ReplayNVAS v3/<scene>/<frame_id> folders to Replay metadata rows.
# 'offset': use frame_id as the 0-based temporal offset within each DSLR sensor.
# 'frame_number': use frame_id as the raw frame_annots.frame_number value.
cfg.dataset.replay_metadata_frame_lookup = 'offset'
# How to interpret ReplayNVAS fixed_rotation `rotations` vectors:
# - 'lookat'  : treat as world-space forward direction; build a full W2C via look-at.
# - 'yaw_only': ignore pitch/roll; project forward onto horizontal plane defined by world_up.
cfg.dataset.fixed_rotation_mode = 'lookat'
# World up axis for look-at construction (ReplayNVAS uses z-up by default).
cfg.dataset.fixed_rotation_world_up = [0.0, 0.0, 1.0]
# Optional input source control for viewpoint-based loaders (near or viewpoint)
cfg.dataset.input_source = 'near'   # 'near' or 'viewpoint'
cfg.dataset.input_viewpoint = 0     # when input_source=='viewpoint', use this viewpoint id
# Frame sampling stride per scene (for ReplayNVAS loaders)
cfg.dataset.frame_stride_train = 20
cfg.dataset.frame_stride_test = 20
# Train crop mode: if true, random crop 2s segments during training; if false, use center crop like test
cfg.dataset.train_random_crop = True
# Optional: when using viewpoint input, randomly choose an input viewpoint
# from the training set per-sample (excluding current target viewpoint).
cfg.dataset.random_input_viewpoint = False
cfg.dataset.random_input_viewpoint_eval = False
cfg.dataset.train_viewpoints = ''
cfg.dataset.val_viewpoints = []
cfg.dataset.use_metadata = False
cfg.dataset.metadata_file = 'metadata_v2.json'
cfg.dataset.frame_scope = ''
cfg.dataset.pose_file = 'images.txt'
cfg.dataset.audio_subdir = ''
cfg.dataset.source_audio_file = 'source.wav'
cfg.dataset.ignore_viewpoints = []

# Optional custom scene split override (for ReplayNVAS scene-based splits)
# If these lists are non-empty, dataset loaders may use them instead of defaults
cfg.dataset.train_scenes = []
cfg.dataset.val_scenes = []
cfg.dataset.test_scenes = []

# Optional bandpass preprocessing for datasets
cfg.dataset.bandpass = CN()
cfg.dataset.bandpass.enable = False
cfg.dataset.bandpass.low_hz = 150.0
cfg.dataset.bandpass.high_hz = -1.0  # use Nyquist-1 when <=0
cfg.dataset.bandpass.order = 5

# NVAS metadata alignment (ReplayNVAS)
cfg.dataset.use_metadata_v2 = False
cfg.dataset.metadata_stride_train = 3
cfg.dataset.metadata_stride_eval = 3
cfg.dataset.cam_ids = []  # optional explicit camera id list (e.g., [1,2,3,4,5,6,7,8]); empty uses repo defaults

cfg.dataset.train = CN()

cfg.dataset.train.sampler = ''
cfg.dataset.train.drop_last = True
cfg.dataset.train.shuffle = True

cfg.dataset.test = CN()
cfg.dataset.test.sampler = ''
cfg.dataset.test.batch_sampler = ''
cfg.dataset.test.drop_last = False
cfg.dataset.test.shuffle = False
cfg.dataset.audio_len_sec = 1.0
# preprocessing
cfg.preprocessing = CN()
cfg.preprocessing.audio = CN()
cfg.preprocessing.audio.sampling_rate = 16000
cfg.preprocessing.audio.max_wav_value = 32768
cfg.preprocessing.stft = CN()
cfg.preprocessing.stft.filter_length = 1024
cfg.preprocessing.stft.hop_length = 160
cfg.preprocessing.stft.win_length = 1024
cfg.preprocessing.mel = CN()
cfg.preprocessing.mel.n_mel_channels = 64
cfg.preprocessing.mel.mel_fmin = 0
cfg.preprocessing.mel.mel_fmax = 8000
cfg.preprocessing.mel.freqm = 0
cfg.preprocessing.mel.timem = 0
cfg.preprocessing.mel.blur = False
cfg.preprocessing.mel.mean = -4.63
cfg.preprocessing.mel.std = 2.74
cfg.preprocessing.mel.target_length = 1024



# model
cfg.model = CN()
cfg.model.file = ''
cfg.model.resume_path = ''
cfg.model.joint_emb_dim = 512
cfg.model.pretrained_encoder = ''
cfg.model.model_type = 'full'
cfg.model.render_type = 'base'
cfg.model.use_visual = True
cfg.model.use_cam_rotation = False
cfg.model.sh_rand_init_std = 0.01
cfg.model.use_groupnorm = False
cfg.model.sh_degree = 2
cfg.model.use_stereo_cues = False
cfg.model.diff_use_inv_distance = False
cfg.model.diff_use_side_mag = False
cfg.model.use_freq_atten = False
cfg.model.freq_atten_alpha = 1.0
cfg.model.use_energy_weighted_freq_dist = False
cfg.model.use_point_rotation = False
cfg.model.flip_cam_y_for_sh = False
cfg.model.xyz_anchor_to_scene = False
cfg.model.xyz_anchor_mode = 'scene_mean'  # scene_mean | viewpoint
cfg.model.xyz_anchor_viewpoint = 0
cfg.model.xyz_anchor_radius = 0.0
cfg.model.xyz_init_std = 10.0
# Optional: initialize Audio3DGS _xyz from ReplayNVAS SfM point cloud (COLMAP points3D).
# When enabled, this overrides the default random init (and skips xyz_anchor_to_scene).
cfg.model.xyz_init_from_sfm = False
cfg.model.sfm_points_file = 'cam_imags/sparse/0/points3D.bin'
cfg.model.sfm_points_sample = 'random'  # 'random' (with replacement) or 'repeat'
cfg.model.sfm_points_jitter_std = 0.0   # add N(0, std^2 I) in world units after sampling
cfg.model.use_hopkins_atten = False     
cfg.model.hopkins_learn_params = False
cfg.model.use_pointwise_alpha = False
cfg.model.normalize_world_coords = False
cfg.model.use_geom_phase = False 
cfg.model.head_width = 0.18
cfg.model.sound_speed = 343.0
cfg.model.abs_mag_output = False
cfg.model.max_norm = 10.0
cfg.model.head_radius = 0.09
cfg.model.geom_phase_f_low_hz = 500.0
cfg.model.geom_phase_f_high_hz = 1500.0
cfg.model.use_geom_phase_residual = True
cfg.model.geom_phase_residual_scale_limit = 0.5
cfg.model.geom_phase_residual_alpha = 1.0
cfg.model.mono_mask_activation = 'sigmoid'  # options: sigmoid (default), relu, none (identity)
# Experimental: scene-shared Gaussians for audio GS-only rendering
cfg.model.shared_gs = CN()
cfg.model.shared_gs.num_gaussians = 256
cfg.model.shared_gs.embed_dim = 32
cfg.model.shared_gs.topk = 4
cfg.model.shared_gs.use_rotation = False
cfg.model.shared_gs.use_anisotropic = True
# Gaussian envelope init/constraints (world units; internally normalized if normalize_world_coords=True)
cfg.model.shared_gs.sigma_init = 50.0
cfg.model.shared_gs.sigma_min = 0.05
cfg.model.shared_gs.sigma_max = 500.0
# xyz init std (world units; internally normalized if normalize_world_coords=True)
cfg.model.shared_gs.xyz_init_std = 50.0
cfg.model.shared_gs.visual = CN()
cfg.model.shared_gs.visual.num_anchors = 256
cfg.model.shared_gs.visual.d_model = 64
cfg.model.shared_gs.visual.nhead = 4
cfg.model.shared_gs.visual.query_audio_scale = 1.0
cfg.model.shared_gs.visual.delta_log_g_limit = 0.35
cfg.model.shared_gs.visual.feature_source = 'gaussian'
cfg.model.shared_gs.visual.token_type = 'pooled_patch'
cfg.model.shared_gs.visual.pooled_patch_grid = 4
cfg.model.shared_gs.visual.image_size = 336
cfg.model.shared_gs.visual.image_roots = ['cam_imags/input', 'cam_imags/images']
cfg.model.shared_gs.visual.cache_dir = 'cam_imags/dinov2_cache'
cfg.model.shared_gs.visual.dinov2_model_name = 'dinov2_vitb14_reg'
cfg.model.shared_gs.visual.dinov2_repo_path = ''
cfg.model.shared_gs.visual.dinov2_checkpoint = ''
cfg.model.shared_gs.visual.dinov2_allow_hub = True
cfg.model.visual = CN()
cfg.model.visual.feature_source = 'gaussian'  # 'gaussian' or 'dinov2_anchor'
cfg.model.visual.image_roots = ['cam_imags/input', 'cam_imags/images']
cfg.model.visual.cache_dir = 'cam_imags/dinov2_anchor_cache'
cfg.model.visual.image_size = 336
cfg.model.visual.batch_size = 4
cfg.model.visual.max_images = 0
cfg.model.visual.min_views_per_anchor = 1
cfg.model.visual.dinov2_model_name = 'dinov2_vitb14_reg'
cfg.model.visual.dinov2_repo_path = ''
cfg.model.visual.dinov2_checkpoint = ''
cfg.model.visual.dinov2_allow_hub = True
cfg.model.visual.use_roi_query_conditioning = False
cfg.model.visual.roi_query_scale = 1.0
cfg.model.visual.roi_use_cls_token = True
cfg.model.visual.roi_image_size = 336
cfg.model.visual.vggt_image_size = 224
cfg.model.visual.vggt_model_img_size = 518
cfg.model.visual.vggt_subset_size = 0
cfg.model.visual.vggt_subset_strategy = 'nearest'
cfg.model.visual.vggt_depth = 4
cfg.model.visual.vggt_last_n_outputs = 1
cfg.model.visual.vggt_trainable = False
cfg.model.visual.vggt_use_anchor_cache = True
cfg.model.visual.vggt_anchor_cache_dir = 'cam_imags/vggt_anchor_cache'
# train
cfg.train = CN()

cfg.train.file = 'BaseTrainer'
cfg.train.trainer = 'BaseTrainer'

cfg.train.resume = False
cfg.train.criterion_file = 'BaseCriterion'
cfg.train.body_sample_ratio = 0.5
cfg.train.n_rays = 1024
cfg.train.n_samples = 64
cfg.train.ddim_steps = 200
cfg.train.ep_iter = 500
cfg.train.lr = 1e-4
cfg.train.lr_backbone = 1e-5
cfg.train.lr_decay = 0.9
cfg.train.lr_decay_step = 10
cfg.train.gamma = 0.1
cfg.train.decay_epochs = 1000
cfg.train.weight_decay = 0.0001
cfg.train.grad_clip = 1.0
cfg.train.max_epoch = 1000
cfg.train.batch_size = 1
cfg.train.num_workers = 4
cfg.train.use_log_mag_loss = True
cfg.train.diff_weight = 1.0
cfg.train.enhanced_weight = 0.0
# Optional: visual point-cloud coupling loss for audioGS _xyz regularization.
# When enabled, it adds a small penalty that pulls learned audio 3D points
# towards the COLMAP/3DGS sparse point cloud (e.g., cam_imags/sparse/0/points3D.ply
# or the cached 256_align_points_gs.pkl generated by AV-Cloud).
cfg.train.vis_coupling_weight = 0.0
cfg.train.vis_coupling_type = 'nn'  # 'nn' (one-way) or 'chamfer' (symmetric)
cfg.train.vis_coupling_num_samples = 4096  # 0 => use all _xyz points
cfg.train.vis_coupling_warmup_epochs = 0.0
cfg.train.vis_coupling_points_pkl = 'cam_imags/sparse/0/256_align_points_gs.pkl'
cfg.train.vis_coupling_points_ply = 'cam_imags/sparse/0/points3D.ply'
cfg.train.lr_sh_mult = 5.0
cfg.train.lr_rot_mult = 5.0
cfg.train.lr_xyz_mult = 3.0
cfg.train.lr_alpha_mult = 1.0
cfg.train.freeze_sh_first_frac = 0.0  # fraction of total epochs to freeze SH params (0 disables)
cfg.train.freeze_sh_epochs = 0        # optional explicit epoch count to freeze SH (overrides fraction when >0)
cfg.train.freeze_xyz_after_sh_unfreeze = False  # when true, freeze xyz once SH is unfrozen
cfg.train.use_mr_stft_loss = True
cfg.train.use_random_stft = True
cfg.train.randstft_max_fft = 32768

cfg.train.print_freq = 10
cfg.train.save_every_checkpoint = True
cfg.train.save_interval = 1
cfg.train.save_freq = 10
cfg.train.valiter_interval = 100
cfg.train.val_when_train = False
cfg.train.val_freq = 5
cfg.train.enable_dpam = False
cfg.train.use_image_psnr_loss = False
cfg.train.image_psnr_weight = 0.01
cfg.train.val_freq = 1
cfg.train.duration = 5
cfg.train.phase_loss_weight = 0.1
cfg.train.phase_loss_target = 'lr' 
# test
cfg.test = CN()

cfg.test.save_imgs = True
cfg.test.is_vis = False


def update_config(config, args):
    config.defrost()
    # set cfg using yaml config file
    config.merge_from_file(args.yaml_file)
    # update cfg using args
    config.merge_from_list(args.opts)
    config.freeze()
