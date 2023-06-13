from sahi_coco import Coco, export_coco_as_yolov5
from sahi.utils.file import load_json, save_json
# COCO_DATASET_NAME = 'virat-aerial-156-frames-v2-coco'
# COCO_DATASET_NAME= 'synth-text-dataset'# /Users/mendeza/Documents/projects/synthtiger/synth-text-dataset
# COCO_DATASET_NAME='synth-text-dataset-simple'
# COCO_DATASET_NAME='synth-text-dataset-2k'
# COCO_DATASET_NAME='synth-text-dataset-v2'
# COCO_DATASET_NAME='synth-text-dataset-v3'
# COCO_DATASET_NAME='synth-text-dataset-v3-fix'
COCO_DATASET_NAME='synth-text-dataset-v3-50k'

# /Users/mendeza/Documents/projects/synthtiger/synth-text-dataset/annotations/dataset.json
# coco = Coco.from_coco_dict_or_path(coco_dict_or_path=f'/Users/mendeza/Downloads/{COCO_DATASET_NAME}/annotations/instances_default.json',
#                                    image_dir=f'/Users/mendeza/Downloads/{COCO_DATASET_NAME}/images/0/',
#                                   remapping_dict={i:i-1 for i in [1,2]})
coco = Coco.from_coco_dict_or_path(coco_dict_or_path=f'/Users/mendeza/Documents/projects/synthtiger/{COCO_DATASET_NAME}/annotations/dataset.json',
                                   image_dir=f'/Users/mendeza/Documents/projects/synthtiger/{COCO_DATASET_NAME}/images/')
print(coco.categories)
# export_coco_as_yolov5(
#     output_dir=f'/Users/mendeza/Downloads/{COCO_DATASET_NAME}-yolov5/', 
#     train_coco=coco, 
#     val_coco=coco, 
#     train_split_rate=0.2, 
#     numpy_seed=1,
#     mod_train_dir = f'/run/determined/workdir/{COCO_DATASET_NAME}-yolov5/',
#     mod_val_dir = f'/run/determined/workdir/{COCO_DATASET_NAME}-yolov5/')

export_coco_as_yolov5(
    output_dir=f'/Users/mendeza/Downloads/{COCO_DATASET_NAME}-yolov5/', 
    train_coco=coco, 
    val_coco=coco, 
    train_split_rate=0.2, 
    numpy_seed=1,
    mod_train_dir = f'/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/{COCO_DATASET_NAME}-yolov5/',
    mod_val_dir = f'/mnt/18f3044b-5d9f-4d98-8083-e88a3cf4ab35/{COCO_DATASET_NAME}-yolov5/')