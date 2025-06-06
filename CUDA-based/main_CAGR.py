import os
import psutil
import argparse
from time import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import wn_treecode

time_start = time()

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='input point cloud file name, must have extension xyz/ply/obj/npy')
parser.add_argument('--width_config', type=str, choices=['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'custom'], required=True, help='choose a proper preset width config, or set it as custom, and use --wsmin --wsmax to define custom widths')
parser.add_argument('--wsmax', type=float, default=0.01, help='only works if --width_config custom is specified')
parser.add_argument('--wsmin', type=float, default=0.04, help='only works if --width_config custom is specified')
parser.add_argument('--iters', type=int, default=40, help='number of iterations')
parser.add_argument('--out_dir', type=str, default='results')
parser.add_argument('--cpu', action='store_true', help='use cpu code only')
parser.add_argument('--tqdm', action='store_true', help='use tqdm bar')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


if os.path.splitext(args.input)[-1] == '.xyz':
    points_normals = np.loadtxt(args.input)
    points_unnormalized = points_normals[:, :3]
elif os.path.splitext(args.input)[-1] in ['.ply', '.obj']:
    import trimesh
    pcd = trimesh.load(args.input, process=False)
    points_unnormalized = np.array(pcd.vertices)
elif os.path.splitext(args.input)[-1] == '.npy':
    pcd = np.load(args.input)
    points_unnormalized = pcd[:, :3]
else:
    raise NotImplementedError('The input file must be have extension xyz/ply/obj/npy')

time_preprocess_start = time()

bbox_scale = 1.1
bbox_center = (points_unnormalized.min(0) + points_unnormalized.max(0)) / 2.
bbox_len = (points_unnormalized.max(0) - points_unnormalized.min(0)).max()
points_normalized = (points_unnormalized - bbox_center) * (2 / (bbox_len * bbox_scale))

points_normalized = torch.from_numpy(points_normalized).contiguous().float()
normals = torch.zeros_like(points_normalized).contiguous().float()
# normals2 = torch.zeros_like(points_normalized).contiguous().float()
b = torch.ones(points_normalized.shape[0], 1) * 0.5
mu=torch.zeros_like(points_normalized).contiguous().float()
# rk1 = torch.ones(points_normalized.shape[0], 1) * 0.5
# rk2 = torch.ones(points_normalized.shape[0], 1) * 0.5
# rk3 = torch.ones(points_normalized.shape[0], 1) * 0.5
rk1 = torch.ones_like(points_normalized).contiguous().float() * 0.5
pk = torch.ones_like(points_normalized).contiguous().float() * 0.5
lambda0=0.0
alpha=0.0
N=30
# b2 = torch.ones(points_normalized.shape[0]*3, 1) * 0.5
widths = torch.ones_like(points_normalized[:, 0])    # we support per-point smoothing width, but do not use it in experiments

if not args.cpu:
    points_normalized = points_normalized.cuda()
    normals = normals.cuda()
    # normals2 = normals2.cuda()
    rk1=rk1.cuda()
    pk = pk.cuda()
    b = b.cuda()
    widths = widths.cuda()

wn_func = wn_treecode.WindingNumberTreecode(points_normalized)

preset_widths = {
    'l0': [0.002, 0.016],   # [0.002, 0.016]: noise level 0, used for uniform, noise free points in the paper
    'l1': [0.01, 0.04],     # [0.01, 0.04]: noise level 1, used for real scans in the paper
    'l2': [0.02, 0.08],     # [0.02, 0.08]: noise level 2, used for sigma=0.25% in the paper
    'l3': [0.03, 0.12],     # [0.03, 0.12]: noise level 3, used for sigma=0.5% in the paper
    'l4': [0.04, 0.16],     # [0.04, 0.16]: noise level 4, used for sigma=1% in the paper
    'l5': [0.05, 0.2],      # [0.05, 0.2]: noise level 5, used for sparse points and 3D sketches in the paper
    'custom': [args.wsmin, args.wsmax],
}

wsmin, wsmax = preset_widths[args.width_config]
assert wsmin <= wsmax

print(f'[LOG] You are using width config {args.width_config} width wsmin = {wsmin}, wsmax = {wsmax}')


time_iter_start = time()
if wn_func.is_cuda:
    torch.cuda.synchronize(device=None)
with torch.no_grad():
    args.iters=40
    bar = tqdm(range(args.iters)) if args.tqdm else range(args.iters)
    width_scale = wsmin + (0.5) * (wsmax - wsmin)
    
    # rk1=(wn_func.forward_A2T1(b, widths * width_scale)+wn_func.forward_A2T2(b, widths * width_scale)+wn_func.forward_A2T3(b, widths * width_scale))
    rk1=(wn_func.forward_A2T1(b, widths * width_scale)+wn_func.forward_A2T2(b, widths * width_scale)+wn_func.forward_A2T3(b, widths * width_scale))
    # rk1=b
    pk=rk1
    for i in bar:
        width_scale = wsmin + ((args.iters-1-i) / ((args.iters-1))) * (wsmax - wsmin)
        # width_scale = args.wsmin + 0.5 * (args.wsmax - args.wsmin) * (1 + math.cos(i/(args.iters-1) * math.pi))
        if i<=N:
        # # grad step
            A_mu1 = -wn_func.forward_A21(normals, widths * width_scale)
            A_mu2 = -wn_func.forward_A22(normals, widths * width_scale)
            A_mu3 = -wn_func.forward_A23(normals, widths * width_scale)
        #print(A_mu.shape)
            AT_A_mu =( wn_func.forward_A2T1(A_mu1, widths * width_scale)+wn_func.forward_A2T2(A_mu2, widths * width_scale)+wn_func.forward_A2T3(A_mu3, widths * width_scale))
        # AT_A_mu = wn_func.forward_AT(A_mu, widths * width_scale)
            r = (wn_func.forward_A2T1(b, widths * width_scale)+wn_func.forward_A2T2(b, widths * width_scale)+wn_func.forward_A2T3(b, widths * width_scale) - AT_A_mu-lambda0*normals)
            A_r = -(wn_func.forward_A21(r, widths * width_scale)+wn_func.forward_A22(r, widths * width_scale)+wn_func.forward_A23(r, widths * width_scale))
        # A_r = wn_func.forward_A(r, widths * width_scale)
            alpha = (r * r).sum() / (A_r * A_r).sum()
        
            normals = normals + alpha * r
            if i==N:
                A_mu1 = -wn_func.forward_A21(normals, widths * width_scale)
                A_mu2 = -wn_func.forward_A22(normals, widths * width_scale)
                A_mu3 = -wn_func.forward_A23(normals, widths * width_scale)
                AT_A_mu =( wn_func.forward_A2T1(A_mu1, widths * width_scale)+wn_func.forward_A2T2(A_mu2, widths * width_scale)+wn_func.forward_A2T3(A_mu3, widths * width_scale))
                rk1=(wn_func.forward_A2T1(b, widths * width_scale)+wn_func.forward_A2T2(b, widths * width_scale)+wn_func.forward_A2T3(b, widths * width_scale))- AT_A_mu-lambda0*normals
                pk=rk1
        else:
            A_pk1 = -wn_func.forward_A21(pk, widths * width_scale)
            A_pk2 = -wn_func.forward_A22(pk, widths * width_scale)
            A_pk3 = -wn_func.forward_A23(pk, widths * width_scale)
            AT_A_pk1 = wn_func.forward_A2T1(A_pk1, widths * width_scale)
            AT_A_pk2 = wn_func.forward_A2T2(A_pk2, widths * width_scale)
            AT_A_pk3 = wn_func.forward_A2T3(A_pk3, widths * width_scale)
            # alpha2= ((rk1 * rk1).sum()/((AT_A_pk3*pk)+(AT_A_pk2*pk)+(AT_A_pk1*pk)+lambda0*pk*pk).sum())
            alpha2= ((rk1 * rk1).sum()/((A_pk3*A_pk3).sum()+(A_pk2*A_pk2).sum()+(A_pk1*A_pk1).sum()+lambda0*(pk*pk).sum()))
            # 请勿使用alpha2；
            alpha= ((rk1 * rk1).sum()/(((AT_A_pk3*pk)+(AT_A_pk2*pk)+(AT_A_pk1*pk)).sum()+lambda0*(pk*pk).sum()))
            # print(alpha/alpha2)
            # print(torch.linalg.norm(rk1))
            normals = normals+alpha*pk
            rk20=rk1-alpha*(AT_A_pk1)-alpha*(AT_A_pk2)-alpha*(AT_A_pk3)-alpha*lambda0*pk
            # rk20=rk1+alpha*(AT_A_pk1)+alpha*(AT_A_pk2)+alpha*(AT_A_pk3)-alpha*lambda0*pk
            beta= ((rk20 * rk20).sum()) / ((rk1 * rk1).sum())
            pk=rk20+beta*pk
            rk1=rk20

        # # WNNC step
        # out_normals = wn_func.forward_G(normals, widths * width_scale)

        # # rescale
        # out_normals = F.normalize(out_normals, dim=-1).contiguous()
        # normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
        # normals = out_normals.clone() * normals_len
        
#         # # WNNC step
#         # out_normals = wn_func.forward_G(normals, widths * width_scale)

#         # # rescale
#         # out_normals = F.normalize(out_normals, dim=-1).contiguous()
#         # normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
#         # normals = out_normals.clone() * normals_len
#         # out_normals = F.normalize(normals, dim=-1).contiguous()

# # WNNC step
# WNNC step
print(torch.linalg.norm(r))
print(torch.linalg.norm(rk1))
bar2 = tqdm(range(8))
for i in bar2:
    out_normals = wn_func.forward_G(normals, widths * width_scale)

# rescale
    out_normals = F.normalize(out_normals, dim=-1).contiguous()
    normals_len = torch.linalg.norm(normals, dim=-1, keepdim=True)
    normals = out_normals.clone() * normals_len

        
out_normals =normals
# print(np.linalg.norm(normals))
if wn_func.is_cuda:
    torch.cuda.synchronize(device=None)
time_iter_end = time()
print(f'[LOG] time_preproc: {time_iter_start - time_preprocess_start}')
print(f'[LOG] time_main: {time_iter_end - time_iter_start}')

with torch.no_grad():
    out_points_normals = np.concatenate([points_unnormalized, out_normals.detach().cpu().numpy()], -1)
    np.savetxt(os.path.join(args.out_dir, os.path.basename(args.input)[:-4] + f'.xyz'), out_points_normals)

process = psutil.Process(os.getpid())
mem_info = process.memory_info()    # bytes
mem = mem_info.rss
if wn_func.is_cuda:
    gpu_mem = torch.cuda.mem_get_info(0)[1]-torch.cuda.mem_get_info(0)[0]
    mem += gpu_mem
print('[LOG] mem:', mem / 1024/1024)     # megabytes
