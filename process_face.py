import os
import euler
import argparse
import numpy as np
import sys
import cv2
sys.path.append("/opt/tiger/moco")
from fermion_core_thrift import *
from base_thrift import *

def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map

def preprocess(np_boxes: list[float]):
    np_boxes = np.array(np_boxes)
    np_boxes = np_boxes.reshape(-1, 6).tolist()
    preprocessed_boxes = []
    for box in np_boxes:
        # 分离类别ID和边界框坐标以及分数
        clsid, x1, y1, x2, y2, score = int(box[5]), box[0], box[1], box[2], box[3], box[4]
        # 将边界框坐标打包成一个列表
        bbox = [x1, y1, x2, y2]
        # 将信息添加到新列表中
        preprocessed_boxes.append([clsid, score, bbox])
    return preprocessed_boxes
    
def draw_box(im, np_boxes, labels, threshold=0.5):
    """
    Args:
        im (PIL.Image.Image): PIL image
        np_boxes (np.ndarray): shape:[N,6], N: number of box,
                               matix element:[class, score, x_min, y_min, x_max, y_max]
        labels (list): labels:['class1', ..., 'classn']
        threshold (float): threshold of box
    Returns:
        im (PIL.Image.Image): visualized image
    """
    np_boxes = preprocess(np_boxes)
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    clsid2color = {}
    color_list = get_color_map_list(len(labels))
    # print("np_boxes", np_boxes)
    np_boxes = [
        box for box in np_boxes
        if box[1] > threshold and box[0] > -1
    ]
    # np_boxes = np_boxes[expect_boxes, :]

    for dt in np_boxes:
        clsid, bbox, score = int(dt[0]), dt[2], dt[1]
        if clsid not in clsid2color:
            clsid2color[clsid] = color_list[clsid]
        color = tuple(clsid2color[clsid])
        color = (22,222,22)

        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            # print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
            #       'right_bottom:[{:.2f},{:.2f}]'.format(
            #           int(clsid), score, xmin, ymin, xmax, ymax))
            # draw bbox
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=4,
                fill=color)
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)

        # draw label
        text = "{} {:.4f}".format(labels[clsid], score)
        tw, th = draw.textsize(text)
        draw.rectangle(
            [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))
    return im

def InferImages(file_content, client):
    req = InferRequest()
    t = Tensor()
    t.dtype = DataType.STRING
    t.shape = []
    t.str_data = [file_content]
    req.input = [TensorSet(tensors={"image": t})]
    resp = client.Infer(req)
    # print(resp)
    return resp.output[0]  # batch=1, so index 0 is the final result

client = euler.Client(FermionCore, 'sd://tns.cv.facedet_offline.service.sg1', timeout=5000)
import multiprocessing
import uuid
import cv2
import numpy as np
from multiprocessing import Pool

root = "/mnt/bn/data-tns-live-llm/leon/experiments/llm/face/second_stage_train_imgs_2million"
target_path = "/mnt/bn/data-tns-live-llm/leon/experiments/llm/face/cropped_second_stage_imgs_2million"
if not os.path.exists(target_path): os.mkdir(target_path)
# 定义处理单个图像的函数
def process_image(data):
    path1 = os.path.join(root, str(data["room_id"]), data["object1"])
    path2 = os.path.join(root, str(data["room_id"]), data["object2"])
    target_path1 = os.path.join(target_path, str(data["room_id"]), data["object1"])
    target_path2 = os.path.join(target_path, str(data["room_id"]), data["object2"])
    imgs1 = os.listdir(path1)
    imgs2 = os.listdir(path2)
    os.makedirs(target_path1, exist_ok=True)
    os.makedirs(target_path2, exist_ok=True)
    for path in imgs1:
        try:
            with open(os.path.join(path1, path), "rb") as f:
                img = f.read()
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            output_tensor = InferImages(img, client)
            np_bbox = np.array(output_tensor.tensors['prob'].float_data)
            bbox_shape = output_tensor.tensors['prob'].shape
            np_bbox = np_bbox.reshape(bbox_shape)
            if len(np_bbox) != 6:
                continue
            if sum(np_bbox) == 0:
                continue
            x1, y1, x2, y2 = np_bbox[:4].astype(int)
            from PIL import Image, ImageDraw
            img_np = np.frombuffer(img, dtype=np.uint8)  # 首先将字节数据转换为NumPy数组
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # 然后解码图像
            cropped_img = img[y1:y2, x1:x2]
            # 保存裁剪后的图像
            cv2.imwrite(os.path.join(target_path1, path), cropped_img)
        except Exception as e:
            print(f"处理图像出错: {path}, 错误: {e}")

    for path in imgs2:
        try:
            with open(os.path.join(path1, path), "rb") as f:
                img = f.read()
            if img is None:
                raise ValueError(f"无法读取图像: {path}")
            output_tensor = InferImages(img, client)
            np_bbox = np.array(output_tensor.tensors['prob'].float_data)
            bbox_shape = output_tensor.tensors['prob'].shape
            np_bbox = np_bbox.reshape(bbox_shape)
            if len(np_bbox) != 6:
                continue
            if sum(np_bbox) == 0:
                continue
            x1, y1, x2, y2 = np_bbox[:4].astype(int)
            from PIL import Image, ImageDraw
            img_np = np.frombuffer(img, dtype=np.uint8)  # 首先将字节数据转换为NumPy数组
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # 然后解码图像
            cropped_img = img[y1:y2, x1:x2]
            # 保存裁剪后的图像
            cv2.imwrite(os.path.join(target_path2, path), cropped_img)
        except Exception as e:
            print(f"处理图像出错: {path}, 错误: {e}")

# 使用多进程池来并行处理图像
def parallel_process_images(paths):
    with Pool() as pool:
        # 使用pool.map来并行处理每个图像
        results = pool.map(process_image, paths)
    return results
# 假设 imgs_dic 是一个包含图像文件路径的字典
# paths = os.listdir(imgs_dic)
# 调用函数开始多进程处理
import json
with open("/mnt/bn/data-tns-live-llm/leon/experiments/llm/face/processed_pair_face_2m.json", "r") as f:
    data = json.loads(f.read())

results = parallel_process_images(data)