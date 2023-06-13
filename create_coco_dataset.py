from pathlib import Path
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
import pandas as pd
import string
from PIL import Image
from sahi.utils.file import load_json, save_json

def get_annotations():
    '''
    '''
    return

if __name__ == '__main__':
    # DATASET_NAME='synth-text-dataset'
    # DATASET_NAME='synth-text-dataset-simple'
    # DATASET_NAME='synth-text-dataset-2k'
    # DATASET_NAME='synth-text-dataset-v2'
    # DATASET_NAME='synth-text-dataset-v3'
    # DATASET_NAME='synth-text-dataset-v3-fix'
    DATASET_NAME='synth-text-dataset-v3-50k'
    IMG_DIR = f'/Users/mendeza/Documents/projects/synthtiger/{DATASET_NAME}/images/'
    ANN_DIR = f'/Users/mendeza/Documents/projects/synthtiger/{DATASET_NAME}/annotations_csv/'
    img_ps = list(Path(IMG_DIR).glob('*.jpg'))
    print(len(img_ps))
    # return
    anns = [str(i) for i in list(Path(ANN_DIR).glob('*.csv'))]
    # print(anns)
    df = pd.concat([pd.read_csv(i) for i in anns],ignore_index=True)
    ind_2_class = {ind:i for ind,i in enumerate(string.ascii_lowercase+string.ascii_uppercase + string.digits)}
    class_2_ind = {v:k for k,v in ind_2_class.items()}
    # print(df['image_key'].value_counts().index.tolist())
    # imgs = df['image_key'].value_counts().index.tolist()
    # imgs = [Path(i).name for i in imgs]
    # print(imgs)
    # print(width,height)
    # print(coco_image)
    # init coco object
    coco = Coco()
    # append categories
    for category_id, category_name in ind_2_class.items():
        coco.add_category(
            CocoCategory(id=int(category_id), name=category_name)
        )
    print(coco)
    for image_ind,(name,group) in enumerate(df.groupby("image_key")):
        print(image_ind+1,name)
        img_name = Path(name).name
        width, height = Image.open(Path(IMG_DIR).parent / name).size
        coco_image = CocoImage(file_name=img_name, height=height, width=width)
        for ind,row in group.iterrows():
            # print(row)
            x,y,w,h = row['bboxes_x'], row['bboxes_y'],row['bboxes_w'],row['bboxes_h']
            class_id = row['class_id']
            class_name = row['class']
            coco_annotation = CocoAnnotation(
                bbox=[x,y,w,h],
                category_id=int(class_id),
                category_name=class_name,
                image_id=image_ind+1,
            )
            # if coco_annotation.area > 0:
            coco_image.add_annotation(coco_annotation)
            # break
        coco.add_image(coco_image)
        # print(coco_image.annotations)
    result = coco.split_coco_as_train_val(train_split_rate=0.2)

    # train_json_path = Path(ANN_DIR) / "train_rgb.json"
    # val_json_path = Path(ANN_DIR) / "val_rgb.json"
    # save_json(data=result["train_coco"].json, save_path=train_json_path)
    save_json(data=coco.json, save_path=f'/Users/mendeza/Documents/projects/synthtiger/{DATASET_NAME}/annotations/dataset.json')
    # coco
    # save_json(data=result["val_coco"].json, save_path=val_json_path)
        # break
    # for ind,row in df.iterrows():
    #     print(ind+1)
    #     print(row)
    #     img_name = Path(row['image_key']).name
    #     width, height = Image.open(Path(IMG_DIR) / img_name).size
    #     coco_image = CocoImage(file_name=img_name, height=height, width=width)
    #     print(coco_image)
    #     x,y,w,h = row['bboxes_x'], row['bboxes_y'],row['bboxes_w'],row['bboxes_h']
    #     class_id = row['class_id']
    #     class_name = row['class']
    #     coco_annotation = CocoAnnotation(
    #             bbox=[x,y,w,h],
    #             category_id=int(class_id),
    #             category_name=class_name,
    #         )
    #     if coco_annotation.area > 0:
    #         coco_image.add_annotation(coco_annotation)
    #     print(coco_image)
    #     coco.add_image(coco_image)
    #     break

