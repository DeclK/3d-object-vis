from viewer.viewer import Viewer
from pathlib import Path
from scenedataset import SceneDataset
from tqdm import tqdm

def visualization(dataset, class_list, vis_num, offscreen=False):
    for idx in tqdm(range(vis_num)):
        vi = Viewer(box_type="OpenPCDet", bg = (255,255,255), offscreen=offscreen)

        data = dataset[idx]
        points = data['points']
        boxes = data['pred_boxes']
        box_info = data['pred_name']

        vi.add_points(points[:,0:3],
                    radius = 3,
                    scatter_filed = points[:,2],
                    alpha=1,
                    del_after_show = True,
                    add_to_3D_scene = True,
                    add_to_2D_scene = False,
                    color_map_name = "Greys")


        # add pred_boxes
        vi.add_3D_boxes(boxes=boxes[:,0:7],
                        box_info=box_info,
                        color="green",
                        mesh_alpha = 0.1,  # 表面透明度
                        show_corner_spheres = True,    # 是否展示顶点上的球
                        caption_size=(0.1,0.1),
                        add_to_2D_scene=True,
                        )

        gt_boxes = data['gt_boxes']
        gt_name = data['gt_name']
        class_mask = [i.split(':')[0] in class_list for i in gt_name]
        gt_boxes = gt_boxes[class_mask,:]
        gt_name = gt_name[class_mask]

        # add gt_boxes
        vi.add_3D_boxes(gt_boxes, 
                        color='red',
                        box_info=gt_name,
                        caption_size=(0.1,0.1),
                        is_label=True
                        )


        # set calib matrix
        image = data['image']
        V2C = data['calib']['V2C']
        P2 = data['calib']['P2']
        R0_rect = data['calib']['R0_rect']

        vi.add_image(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)
        vi.set_r0_rect_mat(R0_rect)

        save_path = Path(f'./output')
        save_img_path = save_path / 'img'
        save_velo_path = save_path / 'velo'
        save_img_path.mkdir(parents=True, exist_ok=True)
        save_velo_path.mkdir(parents=True, exist_ok=True)

        save_name = save_img_path / f'{idx:0>4d}.png'
        vi.show_2D(save_name=str(save_name) if offscreen else None)
        save_name = save_velo_path / f'{idx:0>4d}.png'
        vi.show_3D(save_name=str(save_name) if offscreen else None)

if __name__ == '__main__':
    data_root = Path(__file__).parent / 'kitti_data'
    result_path = Path(r'D:\VS_Project\3D-Detection-Tracking-Viewer\kitti_data\output\eval\result.pkl')
    dataset = SceneDataset(data_root, result_path)
    class_list = ['Car', 'Pedestrian', 'Cyclist']
    visualization(dataset, class_list, vis_num=100, offscreen=True)