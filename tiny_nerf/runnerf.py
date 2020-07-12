import os, sys
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import SimpleITK as sitk
import pickle
from pathlib import Path

tf.compat.v1.enable_eager_execution()


def savepkl(arr, filename):
    with open(filename, 'wb') as f:
        pickle.dump(arr, f)


def loadpkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')
    parser.add_argument("--gpu", type=str, default="0",
                        help='gpu id')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    parser.add_argument("--ignore_bg", action='store_true',
                        help='do not sample background points while trainning')

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=1,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_alpha", action='store_true',
                        help='output ch = 2, one for alpha, one for intensity.')
    parser.add_argument("--use_bounds", action='store_true',
                        help='use bounds for each ray (for mri slice)')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none; 1 for gaussian')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # mri flags
    parser.add_argument("--pose", type=str, default='0507-rest',
                        help='options : 0409-rest / 0507-n / 0507-rest')
    parser.add_argument("--multivol", action='store_true',
                        help='if training data contains multiple volumes')
    parser.add_argument("--bonemask", action='store_true',
                        help='if load bone mask')
    parser.add_argument("--classes", type=int, default=1,
                        help='if bonemask is true, how many labels')
    parser.add_argument("--downsample", type=int, default=5,
                        help='downsample synthetic multi volume data')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** tf.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = tf.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return tf.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tf.math.sin, tf.math.cos],
        'gaussian_scale': 5.0,
        'embedding_size': 256
    }
    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


def init_model(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    D = args.netdepth
    W = args.netwidth

    relu = tf.keras.layers.ReLU()

    def dense(W, act=relu):
        return tf.keras.layers.Dense(W, activation=act)

    output_ch = args.classes + 1
    if args.use_alpha:
        output_ch = 2

    inputs = tf.keras.Input(shape=input_ch)
    inputs.set_shape([None, input_ch])
    outputs = inputs

    skips = [4]
    for i in range(D):
        outputs = dense(W)(outputs)
        if i in skips:
            outputs = tf.concat([inputs, outputs], -1)

    outputs = dense(output_ch, act=None)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model, embed_fn


def pt2voxel(pts_batch, pts_batch_voxel, normalize_min_range, args):
    N_samples = args.N_samples
    normalize_min = normalize_min_range[0]
    normalize_range = normalize_min_range[1]

    pts_batch_flat = pts_batch.reshape([-1, pts_batch.shape[-1]])
    pts_batch_voxel_flat = pts_batch_voxel.reshape([-1, pts_batch_voxel.shape[-1]])

    N_voxels = pts_batch_flat.shape[0]
    pts_batch_voxel_spacing = pts_batch_voxel_flat[:, :3]  # N_voxels, 3
    pts_batch_voxel_transmat = pts_batch_voxel_flat[:, 3:].reshape(N_voxels, 4, 4)

    # sample in voxel and transform
    # pts_batch_sample = []
    # for pt, spacing, trans_mat in zip(pts_batch_flat, pts_batch_voxel_spacing, pts_batch_voxel_transmat):
    #     pt_sample = sample_in_voxel(pt, spacing, trans_mat, N_samples)
    #     pts_batch_sample.append(pt_sample)
    # pts_batch_sample = np.array(pts_batch_sample)

    t_vals = np.linspace(0., 1., N_samples + 1)[:-1]  # ignore end point
    pz, px, py = np.meshgrid(t_vals, t_vals, t_vals, indexing='ij')
    sample_pts = np.stack([pz, px, py], -1)  # [N_samples, N_samples, N_samples, 3]
    sample_pts = np.reshape(sample_pts, [-1, 3])
    pts_batch_sample = sample_pts * pts_batch_voxel_spacing[:, None, :] + pts_batch_flat[:, None, :]

    # transform
    ones = np.ones([pts_batch_sample.shape[0], pts_batch_sample.shape[1], 1])
    verts = np.concatenate([pts_batch_sample, ones], -1)
    verts = np.matmul(verts, pts_batch_voxel_transmat.transpose(0, 2, 1))
    pts_batch_sample = verts[..., :3]

    # normalize
    pts_batch_sample = 2. * (pts_batch_sample - normalize_min) / normalize_range - 1

    return pts_batch_sample.astype(np.float32)


def render_rays(embed_fn, network_fn, pts, pts_voxel, args, normalize_min_range):
    pts_sample = pt2voxel(pts, pts_voxel, normalize_min_range, args)

    def raw2outputs(raw):
        # [N_rays, N_samples, C]
        if raw.shape[-1] == 1:
            weighted_intensity = tf.reduce_sum(raw, axis=-2)
            raw_intensity = weighted_intensity
        else:
            # weights = tf.nn.relu(raw[..., 0])
            # intensity = tf.math.sigmoid(raw[..., 1])
            if args.use_alpha:
                weights = raw[..., 0]
                intensity = raw[..., 1]
                weighted_intensity = tf.reduce_sum(weights * intensity, axis=-1)
                weighted_intensity = weighted_intensity[..., None]
                raw_intensity = tf.reduce_sum(intensity, axis=-1)
                raw_intensity = raw_intensity[..., None]
            else:
                weighted_intensity = tf.reduce_sum(raw, axis=[1, 2])  # N_rays, 1
                weighted_intensity = weighted_intensity[..., None]
                raw_intensity = tf.reduce_sum(raw, axis=-2)  # N_rays, C

        return weighted_intensity, raw_intensity

    # Run network
    pts_flat = tf.reshape(pts_sample, [-1, pts_sample.shape[-1]])

    chunk = args.chunk
    raw_flat = tf.concat(
        [network_fn(embed_fn(pts_flat[i:i + chunk])) for i in range(0, pts_flat.shape[0], chunk)], 0)

    raw = tf.reshape(raw_flat, list(pts_sample.shape[:-1]) + [raw_flat.shape[-1]])

    weighted_intensity, raw_intensity = raw2outputs(raw)

    tf.debugging.check_numerics(weighted_intensity, 'weighted density')
    tf.debugging.check_numerics(raw_intensity, 'raw density')
    w_density = tf.reshape(weighted_intensity, list(pts.shape[:-1]) + [weighted_intensity.shape[-1]])
    r_density = tf.reshape(raw_intensity, list(pts.shape[:-1]) + [raw_intensity.shape[-1]])

    return w_density, r_density


def load_tinynerf(name):
    tinynerf_dict = loadpkl(name)
    return tinynerf_dict


def save_tinynerf(nerf_folder=Path("tiny_nerf/nerf_vol2_rest_fg_n1")):
    config = nerf_folder / "config.txt"
    ft_weights_f = nerf_folder / "model.npy"
    range_f = nerf_folder / "normalize_min_range.txt"
    target_nii_f = nerf_folder / "1.nii"
    transform_f = nerf_folder / "transform.txt"

    config_f = open(config)
    parser = config_parser()
    args = parser.parse_args(config_file_contents=config_f.read())

    target_nii = sitk.ReadImage(str(target_nii_f))
    transform = np.loadtxt(str(transform_f)).astype(np.float32)
    normalize_min_range = np.loadtxt(str(range_f)).astype(np.float32)

    hand_vol_dict = {
        "network_weight": np.load(ft_weights_f, allow_pickle=True),
        "args": args,
        "normalize_min_range": normalize_min_range,
        "transform": transform,
        "target_size": target_nii.GetSize(),
        "target_spacing": target_nii.GetSpacing(),
        "target_origin": target_nii.GetOrigin(),
        "target_direction": target_nii.GetDirection(),
    }
    savepkl(hand_vol_dict, str(nerf_folder / "nerf_model.pkl"))


def nerf_init(nerf_pkl="tiny_nerf/nerf_vol2_rest_fg_n1", gpu="0"):
    nerf_pkl = Path(nerf_pkl)
    if not (nerf_pkl / "nerf_model.pkl").exists():
        save_tinynerf(nerf_pkl)

    nerf_dict = load_tinynerf(nerf_pkl / "nerf_model.pkl")
    args = nerf_dict['args']
    args.N_samples = 1

    model, embed_fn = init_model(args)
    print('Reloading weights')
    model.set_weights(nerf_dict['network_weight'])
    normalize_min_range = nerf_dict['normalize_min_range']
    return nerf_dict, args, model, embed_fn, normalize_min_range


def ostu_seg(vol):
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    seg = otsu_filter.Execute(vol)

    return seg


def nda2vol(nda, spacing, target_origin, target_direction):
    mri_nii = sitk.GetImageFromArray(nda)
    mri_nii.SetSpacing(spacing)
    mri_nii.SetOrigin(target_origin)
    mri_nii.SetDirection(target_direction)
    return mri_nii


def intensity2label(intensity):
    label = tf.argmax(intensity, axis=-1)  # N
    label = label[..., None]
    return label


def nerf_getvalue(foreground_mask, pts, expand_spacing, nerf_dict, args, model, embed_fn, normalize_min_range,
                  ori_size=None):
    if ori_size is None:
        ori_size = pts.shape[:3]
    assert len(ori_size) == 3

    pts_flat = pts.reshape(-1, pts.shape[-1])
    batch = int(args.N_rand / (args.N_samples ** 3))
    mri = None
    label = None
    for i in range(0, pts_flat.shape[0], batch):

        pts_batch = pts_flat[i:i + batch]

        spacing_tile = np.tile(expand_spacing, list(list(pts_batch.shape[0:-1]) + [1]))  # D,H,W, 3
        transform_tile = np.tile(np.eye(4).flatten(), list(list(pts_batch.shape[0:-1]) + [1]))  # D,H,W, 16
        pts_voxel = np.concatenate([spacing_tile, transform_tile], axis=-1).astype(np.float32)  # D,H,W, 19

        mri_batch, label_batch = render_rays(embed_fn, model, pts_batch, pts_voxel,
                                             args, normalize_min_range)
        # if not args.use_alpha:
        #     label_batch = intensity2label(label_batch)
        if mri is None:
            mri = mri_batch
            label = label_batch
        else:
            mri = np.concatenate([mri, mri_batch], axis=0)
            label = np.concatenate([label, label_batch], axis=0)

    # mri = np.array(mri).reshape(ori_size).transpose(2, 1, 0)
    # label = np.array(label).reshape(ori_size).transpose(2, 1, 0)

    mri_real = np.zeros(ori_size).reshape(-1)
    mri_real[foreground_mask] = np.array(mri).squeeze()
    mri = mri_real
    # mri_nii = nda2vol(mri, expand_spacing, nerf_dict['target_origin'], nerf_dict['target_direction'])

    if args.use_alpha:
        label_real = np.ones(ori_size).reshape(-1) * 0.12
        label_real[foreground_mask] = np.array(label).squeeze()
        label = label_real.reshape(ori_size).transpose(2, 1, 0)
        label_nii = nda2vol(label, expand_spacing, nerf_dict['target_origin'], nerf_dict['target_direction'])
        label_nii = sitk.InvertIntensity(label_nii)
        seg = ostu_seg(label_nii)
        # vectorRadius = (2, 2, 2)
        # kernel = sitk.sitkBall
        # seg = sitk.BinaryMorphologicalClosing(seg, vectorRadius, kernel)
        label_nii = seg
    else:
        label_real_all = np.zeros(ori_size)
        for i in range(args.classes):
            label_real = np.zeros(ori_size).reshape(-1)
            label_real[foreground_mask] = np.array(label[:, i + 1]).squeeze()
            label_i = label_real.reshape(ori_size).transpose(2, 1, 0)
            label_i_nii = nda2vol(label_i, expand_spacing, nerf_dict['target_origin'], nerf_dict['target_direction'])
            seg_nda = sitk.GetArrayFromImage(ostu_seg(label_i_nii)).transpose(2, 1, 0)
            label_real_all[seg_nda == 1] = i + 1

        label = label_real_all.reshape(-1, 1)
        # label_nii = nda2vol(label, expand_spacing, nerf_dict['target_origin'], nerf_dict['target_direction'])

        # label_real = np.zeros(ori_size).reshape(-1)
        # label_real[foreground_mask] = np.array(label).squeeze()
        # label = label_real.reshape(ori_size).transpose(2, 1, 0)
        # label_nii = nda2vol(label, expand_spacing, nerf_dict['target_origin'], nerf_dict['target_direction'])

    return mri, label


from skimage import measure
import trimesh
from mri import HandMRI
from hand_mesh import HandMesh
import config


class TinyNerf:
    def __init__(self, nerf_pkl="/data/new_disk/liyuwei/mano/tiny_nerf/nerf_vol4_rest_fg_n2_muscle/"):
        # nerf_pkl = "../mano/tiny_nerf/nerf_vol4_rest_fg_n2_muscle/"
        # gpu = "2"
        self.nerf_dict, self.args, self.model, self.embed_fn, self.normalize_min_range = nerf_init(nerf_pkl)

    def render_pts(self, pts, inside_mask, vol_size, spacing=[0.5, 0.5, 0.5]):
        mri, label = nerf_getvalue(inside_mask,
                                   pts, spacing,
                                   self.nerf_dict, self.args, self.model, self.embed_fn, self.normalize_min_range,
                                   ori_size=vol_size)
        mri_nii = nda2vol(mri.reshape(vol_size).transpose(2, 1, 0), spacing, self.nerf_dict['target_origin'],
                          self.nerf_dict['target_direction'])
        label_nii = nda2vol(label.reshape(vol_size).transpose(2, 1, 0), spacing, self.nerf_dict['target_origin'],
                            self.nerf_dict['target_direction'])
        return mri, label, mri_nii, label_nii


mri_spacing = [0.5, 0.5, 0.5]
# global_trans = [137.6553, 232.4672, 69.1232]

if __name__ == "__main__":
    # q_lists = "/data/new_disk/liyuwei/mano/tiny_nerf/test/"
    q_lists = "/data/new_disk/liyuwei/video/ev_20200708_151823/"
    nerf_pkl = "../mano/tiny_nerf/nerf_vol4_rest_fg_n2_muscle/"
    gpu = "2"
    tnerf = TinyNerf(nerf_pkl, gpu="2")
    hand_mri = HandMRI()
    hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)

    for i in range(500):
        fname_mriq = q_lists + "{:06d}_mri_vol_q.npy".format(i + 1)
        fname_mri = q_lists + "{:06d}_mri_vol.npy".format(i + 1)

        if not os.path.exists(fname_mriq):
            continue

        fname_size = q_lists + "{:06d}_mri_vol_q_size.npy".format(i + 1)
        pts_vol_size = np.load(fname_size).astype(np.int).reshape(-1)

        theta_mano = np.load(q_lists + "{:06d}_abs_quat.npy".format(i + 1))
        v, j = hand_mesh.set_abs_quat(theta_mano)
        pts_vol = hand_mri.mano_to_vol(v, spacing=mri_spacing)
        assert np.sum(pts_vol.shape[:3] - pts_vol_size) == 0

        mri_pts = np.load(fname_mri)
        mri_pts_q = np.load(fname_mriq)
        # np.savetxt(q_lists + "{:06d}_mri_vol_q.xyz".format(i+1), mri_pts_q)
        # break
        inside_mask = np.ones(mri_pts_q.shape[0]).astype(np.int)
        inside_mask = inside_mask > 0

        mri, label = tnerf.render_pts(mri_pts_q, inside_mask, pts_vol_size, spacing=mri_spacing)
        sitk.WriteImage(mri, q_lists + "{:06d}_mri_vol.nii.gz".format(i + 1))
        sitk.WriteImage(label, q_lists + "{:06d}_mri_vol_label.nii.gz".format(i + 1))
        print(q_lists + "{:06d}_mri_vol.nii.gz".format(i + 1))
