import numpy as np
from PIL import Image
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
import json
import cv2
import torch
import torch.nn.functional as F
import time


# 깊이 이미지와 컬러 이미지를 기반으로 카메라 좌표계 3D point cloud 생성
def depth_image_to_point_cloud(depth_image, color_image, fx, fy, cx, cy, camera2base, mask):

    height, width = depth_image.shape 
    u, v = np.meshgrid(np.arange(width), np.arange(height)) # 이미지 높이와 너비에 대해 좌표그리드 (u,v)생성
    # intrinsic 이용해서 각 픽셀의 3D좌표 계산
    Z = depth_image
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    mask = mask * (depth_image > 0.001) * (depth_image < 1.2) # 깊이가 유효한 범위(0.001 보다 크고 1.2보다 작은)픽셀만 선택할 마스크
    mask = mask > 0
    point_cloud = np.dstack((X, Y, Z)) 
    point_cloud = point_cloud[mask] 
    color = color_image[mask]
    pts = merge_point_clouds(point_cloud, color, camera2base)

    return pts


# 계산된 pc에 동촤 좌표 부여한 후, 제공된 trans matrix을 적용하여 base 좌표계로 변환. -> 이후 특정 z 범위 내의 점만 필터링 -> 색상 정보와 함꼐 merge.
def merge_point_clouds(points, colors, trans):
    column = np.ones((points.shape[0], 1)) # homogeneous 표현
    Tp_Nx4 = np.hstack((points, column)) # 
    Tp_4xN = np.transpose(Tp_Nx4)
    matrix_Nx4 = np.dot(trans, Tp_4xN).T # cam좌표에서 base 좌표로 변환
    matrix_3columns = matrix_Nx4[:, :3] # 

    z_mask = (matrix_3columns[:, 2] > -0.3) *  (matrix_3columns[:, 2] < -0.1)
    merge = np.concatenate((matrix_3columns[z_mask, :], colors[z_mask, :].astype(np.uint8)), axis=1)
    # 필터링된 포인트에 색상정보 포함시켜 최종 결과 반환
    return merge

# 
# 2 개의 소스, 타겟 pc 컬러 정보를 포함한 ICP(Iterative Closet Point) 알고리즘을 통해 align
def coloricp(source, target):
    # 입력 numpy를 Open3D의 포인트 클라우드 객체로 변환
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source[:,:3])  # x, y, z
    src.colors = o3d.utility.Vector3dVector(source[:,3:]/255)  # R, G, B
    
    tar = o3d.geometry.PointCloud()
    tar.points = o3d.utility.Vector3dVector(target[:,:3])  # x, y, z
    tar.colors = o3d.utility.Vector3dVector(target[:,3:]/255)  # R, G, B

    # 3단계로 voxel down sampling을 수행하기 위함
    voxel_radius = [0.02, 0.01, 0.005]
    max_iter = [30, 20, 10]
    # max_iter = [200, 100, 50]
    current_transformation = np.identity(4)

    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = src.voxel_down_sample(radius)
        target_down = tar.voxel_down_sample(radius)

        # 각 다운샘플링 단계마다 normal 추정을 진행
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        
        # 지정된 voxel 반경과 반복 횟수를 적용하여, colored ICP registration 수행
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation # 최종적으로 소스 pc를 타겟에 정합하는 변환 행렬 반환.

    return current_transformation


# depth, image, mask 읽어들여와 pc와 normal 정보 생성.
def get_pts_and_normal(depth_path, image_path, normal_path, extrinsic_file, mask_path):
    fx, fy = 385.86016845703125, 385.3817443847656  # Focal lengths in x and y
    cx, cy = 325.68145751953125, 243.561767578125  # Principal point (image center)
    intrinsic = np.array([[385.86016845703125, 0.0, 325.68145751953125],
                          [0.0, 385.3817443847656, 243.561767578125],
                          [0.0, 0.0, 1.0]])
    # 카메라 내부 파라미터와 카메라-핸드 간의 고정 변환 행렬을 정의.
    camera2hand = np.array([[0.70710678,  0.70710678,   0,  0.008],
                            [-0.70710678, 0.70710678,   0, -0.008],
                            [0,           0,            1,  0.026],
                            [0,           0,            0,  1    ]])
    depth_list = sorted(os.listdir(depth_path))
    image_list = sorted(os.listdir(image_path))
    mask_list = sorted(os.listdir(mask_path))
    f = open(extrinsic_file, 'r') 
    pts = []
    camera2base_list = []
    for i in range(len(depth_list)):
        param = f.readline().split(', ')
        
        # extrinsic 파일에서 각 프레임에 대한 hand2base변환 파라미터(rotation & translation)를 읽어서 변환행렬 구성.
        rot_vec = np.array([float(param[3]), float(param[4]), float(param[5])])
        trans = np.array([float(param[0]), float(param[1]), float(param[2])]).T
        rot_mat = R.from_rotvec(rot_vec).as_matrix()
        hand2base = np.zeros((4,4))
        hand2base[:3,:3] = rot_mat
        hand2base[:3,3] = trans
        hand2base[3,3] = 1

        # cam2base 행렬 계산.
        camera2base = np.dot(hand2base, camera2hand)
        camera2base_list.append(camera2base)
    f.close()
    for i in range(0, len(depth_list)):
        depth = np.load(depth_path + depth_list[i]) / 1000
        image = np.array(Image.open(image_path + image_list[i]))
        mask = np.load(mask_path + mask_list[i])
        point_cloud = depth_image_to_point_cloud(depth, image, fx, fy, cx, cy, camera2base_list[i], mask) # 각 프레임마다 카메라좌표계의 pc 생성.
        pts.append(point_cloud)
    for i in range(len(depth_list)):
        depth = np.load(depth_path + depth_list[i]) / 1000
        save_path = normal_path + depth_list[i]
        cal_normal(depth, camera2base_list[i], save_path) # 각 depth 이미지에 대해 normal 정보를 계산 및 저장
        # least_square_normal(depth, intrinsic, camera2base_list[i], save_path)
    return pts, camera2base_list # 모든 pc와 cam2base 행렬 반환


# (이미 있음)생성된 cam2base 변환 행렬들을 포함, 카메라의 내부 파라미터 정의, 프레임별 변환 정보를 JSON 형식으로 저장
def gen_transforms(camera2base_list, save_path):
    camera = dict()
    camera['fl_x'] = 385.86016845703125
    camera['fl_y'] = 385.3817443847656
    camera['cx'] = 325.68145751953125
    camera['cy'] = 243.561767578125
    camera['w'] = 640
    camera['h'] = 480
    camera['camera_model'] = "OPENCV"
    camera['k1'] = -0.055006977170705795
    camera['k2'] = 0.06818309426307678
    camera['p1'] = -0.0007415282307192683
    camera['p2'] = 0.0006959497695788741
    frames = []
    for i in range(len(camera2base_list)):
        transform_dict = {}
        transform_dict["file_path"] = "images/images_" + str("%04d"%(i+1)) + '.png'
        transform_list = []
        for j in range(4):
            transform_list.append(camera2base_list[i][j, :].tolist())
        transform_dict["transform_matrix"] = transform_list
        frames.append(transform_dict)
    camera['frames'] = frames
    camera_save = json.dumps(camera)
    with open(save_path, "w") as file:
        file.write(camera_save)

### 'colmap/sparse/0/images.txt'
# 각 프레임의 cam2base 변환 행렬 정보 이용 -> 이미지별 정보를 텍스트형식으로 저장.
def gen_image_info(camera2base_list, save_path):
    f = open(save_path, 'w')
    for i in range(len(camera2base_list)):
        line_list = []
        camera2base = camera2base_list[i]
        line_list.append(str(i + 1))
        rvec = R.from_matrix(camera2base[:3,:3]).as_quat().tolist()
        rvec = [rvec[3], rvec[0], rvec[1], rvec[2]] # cam2base 행렬에서 회전 정보를 쿼터니언으로 변환.
        for j in range(len(rvec)):
            line_list.append(str(rvec[j]))
        tvec = camera2base[:3,3].tolist() # translation 벡터
        for j in range(len(tvec)):
            line_list.append(str(tvec[j])) # 쿼터니언 rotation 과 translation 벡터를 한 리스트에 저장.
        line_list.append('1')
        line_list.append('images_' + str("%04d"%(i+1)) + '.png')
        line = ' '.join(line_list) + '\n' + '0 0 0' + '\n'
        f.write(line)

### 'colmap/sparse/0/cameras.txt'
# 카메라 내부 파라미터와 왜곡 계수를 지정된 포맷에 맞게 txt 파일로 저장.
def gen_camera_info(save_path):
    f = open(save_path, 'w')
    line_list = []
    line_list.append('1')
    line_list.append('OPENCV')
    line_list.append('640')
    line_list.append('480')
    line_list.append('385.86016845703125')
    line_list.append('385.3817443847656')
    line_list.append('325.68145751953125')
    line_list.append('243.561767578125')
    line_list.append('-0.05539723485708237')
    line_list.append('0.06696220487356186')
    line_list.append('-0.0005387895507737994')
    line_list.append('0.0007650373736396432')
    line = ' '.join(line_list) + '\n'
    f.write(line)


# depth 이미지로부터 각 픽셀의 normal을 계산. -> 저장 및 시각화 이미지 생성.
def cal_normal(depth, trans, save_path):
    fx, fy = 385.86016845703125, 385.3817443847656  # Focal lengths in x and y
    depth[depth < 0.01] = 1e-5 # depth 값 너무 작으면 안정성 위해 일정값으로 대체.
    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image. gradient 함수 사용해서 depth 이미지의 x,y방향 기울기 계산.
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]

    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    normal_unit = normal_unit.reshape(-1, 3).T
    normal_unit = np.dot(trans[:3,:3], normal_unit).T
    normal_unit = normal_unit.reshape((480, 640, 3))
    np.save(save_path, normal_unit)

    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
    normal_vis = vis_normal(normal_unit)
    save_path_vis = save_path.replace('normals', 'normal_vis').replace('.npy', '.png')
    cv2.imwrite(save_path_vis, normal_vis) 

# depth 이미지의 각 픽셀을 cam좌표계의 3D좌표로 변환
def get_points_coordinate(depth, instrinsic_inv, device="cuda"):
    B, height, width, C = depth.size()
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                           torch.arange(0, width, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv.float(), xyz) # [B, 3, H*W]
    depth_xyz = xyz * depth.view(B, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, height, width)

# depth 이미지로부터 주변 이웃 픽셀 정보를 활용하여 Least square를 통해 normal을 계산
def least_square_normal(depth, intrinsic, trans, save_path):
     # load depth & intrinsic
    H, W = depth.shape
    depth_torch = torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1) # (B, h, w, 1)
    intrinsic_inv_np = np.linalg.inv(intrinsic)
    intrinsic_inv_torch = torch.from_numpy(intrinsic_inv_np).unsqueeze(0) # (B, 4, 4)

    ## step.2 compute matrix A
    # compute 3D points xyz
    points = get_points_coordinate(depth_torch, intrinsic_inv_torch[:, :3, :3], "cpu")
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # An = b
    matrix_a = point_matrix.view(1, 3, 25, H, W)  # (B, 3, 25, HxW)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, HxW, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([1, H, W, 5, 1])

    # dot(A.T, A)
    point_multi = torch.matmul(matrix_a_trans, matrix_a)
    matrix_deter = torch.det(point_multi.to("cpu"))
    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
    # diag matrix to update uninverse
    diag_constant = torch.ones([3], dtype=torch.float32)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(1, H, W, 1, 1)
    # inversible matrix
    inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
    inv_matrix = torch.inverse(inversible_matrix.to("cpu"))

    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm, p=2, dim=3).numpy()
    normal_unit = norm_normalize.reshape(-1, 3).T
    normal_unit = np.dot(trans[:3,:3], normal_unit).T
    normal_unit = normal_unit.reshape((480, 640, 3))


    ## step.4 save normal vector
    # np.save(save_path, norm_normalize_np)
    norm_normalize_vis = (((normal_unit + 1) / 2) * 255)[..., ::-1].astype(np.uint8)
    save_path_vis = save_path.replace('normals', 'normal_vis').replace('.npy', '.png')
    cv2.imwrite(save_path_vis, norm_normalize_vis)

###############################################################################################################################################

def gen_pointcloud(depth_path, image_path, mask_path, transforms_save_path):
    """
    transforms.json 파일을 로드하여, 각 프레임에 대해
    depth, image, mask 파일을 읽고 point cloud를 생성.
    각 프레임의 point cloud를 모두 합쳐서 반환.
    """
    # transforms.json 파일 로드
    with open(transforms_save_path, 'r') as f:
        transforms_data = json.load(f)
    
    # 카메라 내부 파라미터 (fl_x, fl_y, cx, cy)는 transforms.json 에 저장되어 있음.
    fx = transforms_data.get("fl_x")
    fy = transforms_data.get("fl_y")
    cx = transforms_data.get("cx")
    cy = transforms_data.get("cy")

    # 각 폴더의 파일 리스트 (sorted)
    depth_list = sorted(os.listdir(depth_path))
    image_list = sorted(os.listdir(image_path))
    mask_list = sorted(os.listdir(mask_path))

    frames = transforms_data['frames']
    all_points = []

    for i, frame in enumerate(frames):
        # 각 프레임의 변환 행렬 (4x4)
        camera2base = np.array(frame["transform_matrix"])

        # 파일 경로 구성 (파일 이름은 정렬된 순서로 대응된다 가정)
        depth_file = os.path.join(depth_path, depth_list[i])
        image_file = os.path.join(image_path, image_list[i])
        mask_file  = os.path.join(mask_path, mask_list[i])

        # 깊이 이미지 로드 (깊이 값은 mm 단위를 m로 변환)
        depth = np.load(depth_file)
        # import pdb;pdb.set_trace()
        
        # 컬러 이미지 로드 (PIL 이미지 -> numpy 배열)
        image = np.array(Image.open(image_file))
        
        # 마스크 로드
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) # grayscale로 read
        # 이제 mask는 2D numpy 형태 (H,W). 픽셀값은 [0,255]
        #mask = (mask>0).astype(np.uint8) # boolean으로 변환.

        
        # 포인트 클라우드 생성
        pts = depth_image_to_point_cloud(depth, image, fx, fy, cx, cy, camera2base, mask)
        all_points.append(pts)
    
    # 모든 프레임의 포인트 클라우드 합치기
    all_points = np.concatenate(all_points, axis=0)
    return all_points


def save_points3D(all_points, save_path):
    """
    생성된 point cloud 배열에 대해, 랜덤 샘플링(전체의 1/8)을 수행,
    각 포인트에 고유 ID를 부여한 후 텍스트 파일(points3D.txt)로 저장.
    저장 포맷은: point_id x y z R G B
    """
    num_points = all_points.shape[0]
    # 랜덤 샘플링 (원하는 비율로 조정 가능)
    idx = np.random.choice(np.arange(num_points), size=num_points // 8, replace=False)
    sampled_points = all_points[idx]
    
    # 포인트 ID 부여 (1부터 시작)
    point_ids = np.arange(1, sampled_points.shape[0] + 1).reshape(-1, 1)
    points_with_id = np.concatenate((point_ids, sampled_points), axis=1)
    
    # 각 행: 정수형(point id) + 3D 좌표(float) + 색상(uint8)
    # fmt 포맷은 데이터 타입에 따라 수정 가능
    fmt = '%d ' + '%.6f ' * 3 + '%d %d %d'
    np.savetxt(save_path, points_with_id, fmt=fmt)


def visualize_points3D_txt(txt_file):
    """
    points3D.txt 파일을 읽어 Open3D를 통해 포인트 클라우드를 시각화하는 함수.
    파일 형식 예시 (ID, X, Y, Z, R, G, B):
        1  0.1234  0.5678  0.9101  255 128 64
        2  0.2345  0.6789  0.2222  220 130 70
        ...
    """
    # 1. 파일 로드 (ID, X, Y, Z, R, G, B 형태)
    data = np.loadtxt(txt_file)  # shape: (N, 7)
    
    # 2. 포인트/컬러 분리 (ID는 인덱스 0번, x=1, y=2, z=3, R=4, G=5, B=6)
    points = data[:, 1:4]  # (X, Y, Z)
    colors = data[:, 4:7]  # (R, G, B)
    
    # 3. Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 색상 범위가 [0,255]이므로 [0,1] 범위로 스케일링
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    # 4. 시각화
    o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    path = '/home/jclee/workspace/data_collection/scene_0001/'
    depth_path = path + 'gt_depths(skipped)/'
    image_path = path + 'gt_images(skipped)/'
    normal_path = path + 'gt_normals(skipped)/'
    mask_path = path + 'gt_boundary_mask(skipped)/' # 
    #extrinsic_file = path + 'hand2base.txt' # ???
    transforms_save_path = path + 'gt_transforms(skipped).json' # -> 요놈을 hand2base로 코드 수정 필요.
    # camera_info_save_path = path + 'colmap/sparse/0/cameras.txt' # ???
    #images_info_save_path = path + 'colmap/sparse/0/images.txt' # ???
    os.makedirs('/home/jclee/workspace/data_collection/scene_0001/point_clouds',exist_ok=True)
    pts_save_path = path + 'point_clouds/points3D(skipped).txt' # ???
    # pts, camera2base_list = get_pts_and_normal(depth_path, image_path, normal_path, extrinsic_file, mask_path)

    #gen_transforms(camera2base_list, transforms_save_path) # 이미 있음
    #gen_image_info(camera2base_list, images_info_save_path) # 없어도 됨
    #gen_camera_info(camera_info_save_path) # 없어도 됨
    
    s_t0 = time.time()
    all_points = gen_pointcloud(depth_path, image_path, mask_path, transforms_save_path)
    e_t0 = time.time()
    elapsed0 = e_t0 - s_t0

    s_t1 = time.time()
    # points3D.txt 파일로 저장
    save_points3D(all_points, pts_save_path)
    e_t1 = time.time()
    elapsed1 = e_t1 - s_t1
    
    print(f"Total points: {all_points.shape[0]}")
    print(f"Points saved to: {pts_save_path}")

    s_t2 = time.time()
    pc_vis_file_path = "/home/jclee/workspace/data_collection/scene_0001/point_clouds/points3D(skipped).txt"
    visualize_points3D_txt(pc_vis_file_path)
    e_t2 = time.time()
    elapsed2 = e_t2 - s_t2

    print(f"Point_Cloud Generating Time: {elapsed0:.2f} seconds")
    print(f"Point_Cloud Saving Time: {elapsed1:.2f} seconds")
    print(f"Point_Cloud Visualizing Time: {elapsed2:.2f} seconds")
