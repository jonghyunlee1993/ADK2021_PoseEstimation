import os
import json
from tqdm import tqdm

from sklearn.model_selection import train_test_split


def cal_center_scale(bbox):
    center = [bbox[2] // 2, bbox[3] // 2]
    scale = max(bbox[2], bbox[3]) * 1.2 / 200

    return center, scale


def cvt_kps_format(kps):
    cvt_kps = []

    for i in range(0, len(kps), 3):
        k = [kps[i], kps[i+1]]
        cvt_kps.append(k)

    return cvt_kps


def make_kps_data(root_path, save_path, json_list, dataset_type):
    data_list = []

    isValidation = 0 if dataset_type == "train" else 1

    print("===> generate \'%s\' dataset json file..." % dataset_type)
    for json_file in tqdm(json_list):
        img_filename = os.path.splitext(json_file)[0] + ".jpg"

        json_file_path = os.path.join(root_path, json_file)
        with open(json_file_path, 'r') as jf:
            json_data = json.load(jf)
        
        bbox = json_data['label_info']['annotations'][0]['bbox']
        kps = json_data['label_info']['annotations'][0]['keypoints']

        if len(kps) != 51:
            continue

        center, scale = cal_center_scale(bbox)
        kps = cvt_kps_format(kps)

        data_dict = {
            "isValidation": isValidation,
            "img_paths": img_filename,
            "joint_self": kps,
            "objpos": center,
            "scale_provided": scale
        }

        data_list.append(data_dict)

    save_json_name = "./" + dataset_type + "_mpii_annotations.json"
    save_json_name = os.path.join(save_path, save_json_name)

    with open(save_json_name, 'w') as save_jf:
        json.dump(data_list, save_jf)

    print("===> generated \'%s\' dataset json file completely!\n" % dataset_type)


def gen_train_test_datasets():
    input_img_path = "./datasets/cow/images"
    gt_img_path = "./datasets/cow/annotations"
    gt_save_path= "./datasets/cow"

    input_imgs = sorted(os.listdir(input_img_path))
    gt_jsons = sorted(os.listdir(gt_img_path))

    train_inputs, val_inputs, train_gts, val_gts = \
        train_test_split(input_imgs, gt_jsons, test_size=0.2, random_state=321)

    val_inputs, test_inputs, val_gts, test_gts = \
        train_test_split(val_inputs, val_gts, test_size=0.5, random_state=321)

    train_gts = sorted(train_gts)
    val_gts = sorted(val_gts)
    test_gts = sorted(test_gts)

    assert len(input_imgs) == len(gt_jsons)
    assert len(train_inputs) == len(train_gts)
    assert len(val_inputs) == len(val_gts)
    assert len(test_inputs) == len(test_gts)   


    print("Total # of data: %d" % len(input_imgs))
    print("Total # of \'train\' data: %d" % len(train_inputs))
    print("Total # of \'validation\' data: %d\n" % len(val_inputs))
    print("Total # of \'test\' data: %d\n" % len(test_inputs))


    make_kps_data(gt_img_path, gt_save_path, train_gts, "train")
    make_kps_data(gt_img_path, gt_save_path, val_gts, "val")
    make_kps_data(gt_img_path, gt_save_path, test_gts, "test")


if __name__ == "__main__":
    gen_train_test_datasets()