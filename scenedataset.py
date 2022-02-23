import numpy as np
import pickle
import matplotlib.image as Image


class SceneDataset:
    def __init__(self, data_root, result_file, gt_info_file=None):
        """
        Args:
            - data_root: kitti data dir
                - velodyne
                - image_2
            - result_file: result.pkl path
            - gt_info_file: kitti_infos_val.pkl path
        """
        self.data_root = data_root
        self.result_file = result_file
        self.points_path = data_root  / 'velodyne'
        self.image_path = data_root  / 'image_2'

        self.pred_list = self.get_pickle(self.result_file)
        self.gt_info_file = gt_info_file
        # 是否有 gt 选框，一般为 kitti_infos_val.pkl
        if gt_info_file is not None: 
            self.val_list = self.get_pickle(self.gt_info_file)
            assert len(self.pred_list) == len(self.val_list), "pred & val don't match"
    
    def __len__(self):
        return len(self.pred_list)
    
    def __getitem__(self, idx:int):
        """
        Return idx sample's batch_dict:
            - points
            - pred_boxes
            - pred_name
            - gt_boxes
            - gt_name
            - calib
            - image
        """
        # 获得样本 id
        data_dict = {}
        frame_id = self.pred_list[idx]['frame_id']
        if self.gt_info_file is not None:
            val_id = self.val_list[idx]['point_cloud']['lidar_idx']
            assert frame_id == val_id, f'pred_id: {frame_id}, val_id: {val_id} not the same'

            # 获得 gt_boxes 及其 name: score 标注
            data_dict['gt_boxes'] = self.val_list[idx]['annos']['gt_boxes_lidar']
            num_boxes = data_dict['gt_boxes'].shape[0]
            name = self.val_list[idx]['annos']['name'][:num_boxes]
            score = [1.0 for i in range(num_boxes)]
            data_dict['gt_name'] = self.name_with_score(name, score)

            # 获得 calib matrix
            data_dict['calib'] = dict(
                Tr_velo_to_cam = self.val_list[idx]['calib']['Tr_velo_to_cam'],
                P2 = self.val_list[idx]['calib']['P2'],
                R0_rect = self.val_list[idx]['calib']['R0_rect']
            )

        # 获得 points & pred_boxes
        velo_file = self.points_path / f'{frame_id}.bin'
        img_file = self.image_path / f'{frame_id}.png'
        points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
        data_dict['points'] = points
        data_dict['pred_boxes'] = self.pred_list[idx]['boxes_lidar']

        # 获得 name: score 标注
        score = self.pred_list[idx]['score']
        name = self.pred_list[idx]['name']
        data_dict['pred_name'] = self.name_with_score(name, score)

        data_dict['image'] = Image.imread(str(img_file))
        return data_dict
    
    @staticmethod
    def get_pickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def name_with_score(name, score):
        """
        将名字和得分以 name: score 的形式，放到一个 array 当中
        """
        assert len(name) == len(score), "name & score don't match"
        ret_list = []
        for i in range(len(name)):
            s = f'{name[i]}: {score[i]:.2f}'
            ret_list.append(s)
        ret_array = np.array(ret_list)
        return ret_array

if __name__ == '__main__':
    from pathlib import Path
    data_root = Path(__file__).parent / 'kitti_data'
    dataset = SceneDataset(data_root)
    for i in range(10):
        dataset[i]