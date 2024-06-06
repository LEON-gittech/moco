import json

with open("/mnt/bn/data-tns-live-llm/leon/experiments/llm/face/processed_pair_face_2m.json", "r") as f:
    data = json.loads(f.read())

import requests
import uuid
import os

root = "/mnt/bn/data-tns-live-llm/leon/experiments/llm/face/second_stage_train_imgs_2m"
if not os.path.exists(f'{root}'): os.mkdir(f'{root}')

def get_img_roomid(data):
    if not os.path.exists(f'{root}/{data["room_id"]}/'): 
        os.mkdir(f'{root}/{data["room_id"]}/')
        os.mkdir(f'{root}/{data["room_id"]}/{data["object1"]}')
        os.mkdir(f'{root}/{data["room_id"]}/{data["object2"]}')
    else: return None
    urls1, urls2 = json.loads(data["urls1"]), json.loads(data["urls2"])
    for url in urls1:
        try:
            response = requests.get(url)
            # 检查请求是否成功
            if response.status_code == 200:
                # 使用uuid生成唯一的文件名
                file_name = str(uuid.uuid4()) + '.jpg'
                # 构建文件保存路径
                file_path = os.path.join(f'{root}/{data["room_id"]}/{data["object1"]}', file_name)
                # 将响应内容写入文件
                with open(file_path, 'wb') as file:
                    file.write(response.content)
            else:
                print('图片下载失败，状态码:', response.status_code)
        except Exception as e:
            print(f'图片下载失败，错误：{e}')

    for url in urls2:
        try:
            response = requests.get(url)
            # 检查请求是否成功
            if response.status_code == 200:
                # 使用uuid生成唯一的文件名
                file_name = str(uuid.uuid4()) + '.jpg'
                # 构建文件保存路径
                file_path = os.path.join(f'{root}/{data["room_id"]}/{data["object2"]}', file_name)
                # 将响应内容写入文件
                with open(file_path, 'wb') as file:
                    file.write(response.content)
            else:
                print('图片下载失败，状态码:', response.status_code)
        except Exception as e:
            print(f'图片下载失败，错误：{e}')
            
import multiprocessing
# 使用多进程池来下载图像
def download_images(data):
    with multiprocessing.Pool(20) as pool:
        pool.map(get_img_roomid, data)

# 调用函数开始多进程下载
download_images(data)