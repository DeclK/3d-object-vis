from viewer.viewer import Viewer
from pathlib import Path
from scenedataset import SceneDataset
from tqdm import tqdm
from utils import *

def visualization(dataset, class_list, vis_num, thres = None, offscreen=False):
    save_path = dataset.data_root / 'output'
    save_img_path = save_path / 'img'
    save_velo_path = save_path / 'velo'

    pbar = tqdm(range(vis_num))
    pbar.set_description('visualize')
    for idx in pbar:
        vi = Viewer(box_type="OpenPCDet", bg = (255,255,255), offscreen=offscreen)

        data = dataset[idx]
        points = data['points']
        boxes = data['pred_boxes']
        box_info = data['pred_name']

        # 是否设置阈值筛选
        if thres is not None:
            thres_mask = [float(i.split(':')[1]) > thres for i in box_info]
            boxes = boxes[thres_mask]
            box_info = box_info[thres_mask]

        vi.add_points(points[:,0:3],
                    radius = 3,
                    color= 'dimgray',
                    # scatter_filed = points[:,2],
                    alpha=1,
                    del_after_show = True,
                    add_to_3D_scene = True,
                    add_to_2D_scene = False)
                    # color_map_name = 'Greys')


        # add pred_boxes
        vi.add_3D_boxes(boxes=boxes[:,0:7],
                        box_info=box_info,
                        color="green",
                        mesh_alpha = 0.1,  # 表面透明度
                        show_corner_spheres = True,    # 是否展示顶点上的球
                        caption_size=(0.1,0.1),
                        add_to_2D_scene=True,
                        )

        # 是否有 gt 选框
        if dataset.gt_info_file is not None:
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
        if data.get('calib', None) is None:
            calib_file = dataset.data_root / 'calib.txt'
            data['calib'] = get_calib(calib_file)
        V2C = data['calib']['Tr_velo_to_cam']
        P2 = data['calib']['P2']
        R0_rect = data['calib']['R0_rect']

        vi.add_image(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)
        vi.set_r0_rect_mat(R0_rect)

        # 如果设置 offscreen 则会保存结果
        if offscreen:
            save_path.mkdir(parents=True, exist_ok=True)
            save_img_path.mkdir(parents=True, exist_ok=True)
            save_velo_path.mkdir(parents=True, exist_ok=True)

        save_name = save_img_path / f'{idx:0>4d}.png'
        vi.show_2D(save_name=str(save_name) if offscreen else None)
        save_name = save_velo_path / f'{idx:0>4d}.png'
        vi.show_3D(save_name=str(save_name) if offscreen else None)
    if offscreen:
        print(f'visualization results are saved to: {save_path}')

if __name__ == '__main__':
    data_root = Path(r'data_05')
    result_path = Path(r'result_05.pkl')
    dataset = SceneDataset(data_root, result_path)
    # for i in range(2):
    #     print_dict(dataset.pred_list[i], content=True)

    class_list = ['Car', 'Pedestrian', 'Cyclist']
    visualization(dataset, 
                  class_list,
                  vis_num=len(dataset), 
                  thres = 0.7,
                  offscreen=True)
    
    concat(data_root)