"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
from collections import OrderedDict,Counter
import numpy as np
from PIL import Image, ImageDraw

from synthtiger import components, layers, templates
import string
import pandas as pd
from pathlib import Path
from synthtiger import components, layers, templates, utils
import cv2
import numpy as np
from PIL import Image

BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "hard_light",
    "soft_light",
    "dodge",
    "divide",
    "addition",
    "difference",
    "darken_only",
    "lighten_only",
]

class Multiline(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.coord_output = config.get("coord_output", True)
        self.count = config.get("count", 1)
        self.corpus = components.BaseCorpus(**config.get("corpus", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.color = components.RGB(**config.get("color", {}))
        self.layout = components.FlowLayout(**config.get("layout", {}))
        self.texture = components.Switch(
            components.BaseTexture(), **config.get("texture", {})
        )
        self.colormap3 = components.GrayMap(**config.get("colormap3", {}))
        self.color2 = components.Gray(**config.get("color2", {}))
        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {}),
        )
        self.fit = components.Fit()
        self.pad = components.Switch(components.Pad(), **config.get("pad", {}))
        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]
            ),
            **config.get("shape", {}),
        )
        self.transform = components.Switch(
            components.Selector(
                [
                    components.Perspective(),
                    components.Perspective(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {}),
        )
        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {}),
        )
    def _postprocess_images(self, images):
        image_layers = [layers.Layer(image) for image in images]
        self.postprocess.apply(image_layers)
        outs = [image_layer.output() for image_layer in image_layers]
        return outs
    def generate(self):
        print("self.count:",self.count )
        texts = [self.corpus.data(self.corpus.sample()) for _ in range(self.count)]
        print("texts: ",len(texts),texts[0] )
        fonts = [self.font.sample() for _ in range(self.count)]
        print("font: ",fonts)
        # color = self.color.data(self.color.sample())
        # char_layers = [layers.TextLayer(char, fonts) for char in chars]
        a = []
        ind = 0
        spaces = []
        charys = []
        for text, font in zip(texts, fonts):
                color = self.color.data(self.color.sample())
                for char in text:
                    charys.append(char)
                    a.append(layers.TextLayer(char, color=color, **font))
                    ind+=1
                a.append(layers.TextLayer(' ', color=color, **font))
                spaces.append(ind)
                charys.append(-1)
                ind+=1
        spaces.append(ind)
        print(spaces)
        text_group = layers.Group(a)

        # print("Text Group: ",list(text_group))
        fg_style = self.style.sample()
        transform = self.transform.sample()
        self.shape.apply(text_group)
        self.layout.apply(text_group)
        self.texture.apply(text_group)
        self.style.apply([text_group], fg_style)
        self.transform.apply(
            [text_group], transform
        )
        self.fit.apply([text_group])
        self.pad.apply([text_group])

        # bg_color = self.color.data(self.color.sample())# use if using layers.RectLayer(fg_image.size,bg_color)
        bg_color = self.color.sample()

        # _,bg_color,_=self.colormap3.sample()
        # bg_color = self.colormap3.data(bg_color)
        # bg_color=(bg_color[0],bg_color[1],bg_color[2],10)# use if using layers.RectLayer(fg_image.size,bg_color)

        # print("bg_color: ",bg_color)
        # bg_color[-1]=125
        fg_image = text_group
        bg_layer = layers.RectLayer(fg_image.size)
        # self.color2.apply(bg_layer)
        self.color.apply([bg_layer], bg_color)
        self.texture.apply([bg_layer])

        for char_layer in text_group:
            char_layer.topleft -= text_group.topleft
        bg_layer.topleft = text_group.topleft

        fg_imageI = text_group.output()
        bg_imageI = bg_layer.output()


        # image = (text_group + bg_layer).output()
        image = _blend_images(fg_imageI,bg_imageI,visibility_check=True)
        # image = self._postprocess_images([image])[0]
        # image = image.astype(np.uint8)
        # print(image.dtype)
        # image_p = Image.fromarray(image)
        # draw = ImageDraw.Draw(image_p)
        bboxes = []
        character_labels = []
        for ind,x in enumerate(text_group):
            if ind not in spaces:
                # print(x.bbox, charys[ind].lower(), text_group.topleft)
                print(x.bbox, charys[ind], text_group.topleft)

                x,y,w,h = x.bbox
                bboxes.append([x,y,w,h])
                # character_labels.append(charys[ind].lower())
                character_labels.append(charys[ind])# keep uppercase AND lowercase

                # draw.rectangle([x,y,x+w,y+h],fill=None,outline=(255,0,0))
        # image = np.array(image_p)
        print("image: ",image.shape)

        label = " ".join(texts)

        data = {
            "image": image,
            "label": label,
            "bboxes":bboxes,
            "character_labels":character_labels
        }

        return data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)
        gt_path = os.path.join(root, "gt.txt")
        self.gt_file = open(gt_path, "w", encoding="utf-8")
        # if self.coord_output:
        #     self.coords_file = open(coords_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        bboxes = data["bboxes"]
        coords = "\t".join([",".join(map(str, map(int, coord))) for coord in bboxes])
        character_labels = data['character_labels']
        # print(Counter(character_labels))
        # to coco
        # c=OrderedDict(character_labels)
        ind_2_class = {ind:i for ind,i in enumerate(string.ascii_lowercase+string.ascii_uppercase + string.digits)}
        class_2_ind = {v:k for k,v in ind_2_class.items()}
        print(ind_2_class)
        counter = {i:0 for i in ind_2_class.values()}
        for c in character_labels:
            counter[c]+=1
        print(counter)
            
        # print(c,c.keys())


        # shard = str(idx // 10000)
        image_key = os.path.join("images", f"{idx}.jpg")
        # Path(os.path.join(root, 'annotations_csv')).parent.mkdir(parents=True, exist_ok=True)

        self.coords_file  = os.path.join(root, 'annotations_csv',"coords_{}.csv".format(idx))
        os.makedirs(os.path.dirname(self.coords_file), exist_ok=True)

        image_path = os.path.join(root, image_key)
        if self.coord_output:
            # self.coords_file.write(f"{image_key}\t{coords}\n")
            df = pd.DataFrame({
                'image_key':[image_key]*len(bboxes),
                'ind': list(range(len(bboxes))),
                'bboxes_x': [i[0] for i in bboxes],
                'bboxes_y': [i[1] for i in bboxes],
                'bboxes_w': [i[2] for i in bboxes],
                'bboxes_h': [i[3] for i in bboxes],
                'class_id':[class_2_ind[i] for i in character_labels],
                'class':character_labels
            })
            df.to_csv(self.coords_file,index=None)
        
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_path, quality=95)

        self.gt_file.write(f"{image_key}\t{label}\n")

    def end_save(self, root):
        self.gt_file.close()
def _check_visibility(image, mask):
    gray = utils.to_gray(image[..., :3]).astype(np.uint8)
    mask = mask.astype(np.uint8)
    height, width = mask.shape

    peak = (mask > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bound = (mask > 0).astype(np.uint8)
    bound = cv2.dilate(bound, kernel, iterations=1)

    visit = bound.copy()
    visit ^= 1
    visit = np.pad(visit, 1, constant_values=1)

    border = bound.copy()
    border[mask > 0] = 0

    flag = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for y in range(height):
        for x in range(width):
            if peak[y][x]:
                cv2.floodFill(gray, visit, (x, y), 1, 16, 16, flag)

    visit = visit[1:-1, 1:-1]
    count = np.sum(visit & border)
    total = np.sum(border)
    return total > 0 and count <= total * 0.1
def _blend_images(src, dst, visibility_check=False):
    blend_modes = np.random.permutation(BLEND_MODES)

    for blend_mode in blend_modes:
        out = utils.blend_image(src, dst, mode=blend_mode)
        if not visibility_check or _check_visibility(out, src[..., 3]):
            break
    else:
        raise RuntimeError("Text is not visible")

    return out