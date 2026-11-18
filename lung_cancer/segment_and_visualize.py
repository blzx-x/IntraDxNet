import cv2
import numpy as np
from linux import kfbReader
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from mmseg.apis import init_model, inference_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from svs_reader import WsiReader
import math
import csv

#生成用于无缝拼接的核
def merge_kernel(patch_size):
    kernel_1 = np.zeros((patch_size, patch_size))
    kernel_2 = np.zeros((patch_size, patch_size))
    for y in range(patch_size):
        for x in range(patch_size):
            distance = abs((x - patch_size // 2))
            value = 1 - distance / (patch_size // 2)
            kernel_1[y, x] = value
            kernel_2[x, y] = value
    return kernel_1*kernel_2

def tensor_to_heatmaps(tensor):
    # 计算每个像素点的分类概率
    probabilities = softmax(tensor, axis=0)

    # 将每个分类的概率映射到 [0, 255] 的灰度值范围
    heatmaps = (probabilities * 255).astype(np.float32)

    # 返回每个分类的热力图列表
    return [heatmap for heatmap in heatmaps]

def softmax(x, axis=None):
    # 计算 softmax
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class wsi_to_patches(Dataset):
    def __init__(self, KFB_file, patch_size, read_scale, kfb_height, kfb_width):
        self.KFB_file = KFB_file
        self.y1_x1_list = []
        for y in range(0, kfb_height-patch_size, patch_size//2):
            for x in range(0, kfb_width-patch_size, patch_size//2):
                self.y1_x1_list.append([y, x])
    
    def __len__(self):
        return len(self.y1_x1_list)
    
    def __getitem__(self, index):
        y1, x1 = self.y1_x1_list[index]
        patch = self.KFB_file.ReadRoi(x1, y1, patch_size, patch_size, read_scale)
        return patch

def run(kfb_csv, read_scale, patch_size, config_path, checkpoint_path, save_path, wsi_type):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model = init_model(config_path, checkpoint_path, 'cuda:0')
    kernel = merge_kernel(patch_size)
    if not os.path.exists(os.path.join(save_path, 'wsi_diagnosis.csv')):
        with open(os.path.join(save_path, 'wsi_diagnosis.csv'), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(['kfb_name', 'kfb_path', 'AI诊断', '面积占比(阴，细支气管腺瘤，粘液腺癌)'])
    
    try:
        kfb_df = pd.read_csv(kfb_csv, encoding='utf-8')
    except:
        kfb_df = pd.read_csv(kfb_csv, encoding='gbk')
    
    # 读取 kfb 文件
    kfb_pathes = kfb_df['KFB_path']
    
    for kfb_index in range(len(kfb_pathes)):
        try:
            kfb_path = kfb_pathes[kfb_index]
            kfb_path = kfb_path.replace('\\\\SHARE_HOST\\share', '/data/wsi').replace('\\', '/')
            kfb_name = kfb_path.split('/')[-1]
            if wsi_type == 'kfb':
                KFB_file = kfbReader.reader()
            elif wsi_type =='svs':
                KFB_file = WsiReader()
            else:
                print('WSI类型错误')
                break
            
            try:
                KFB_file.ReadInfo(kfb_path, read_scale, True)
            except Exception as e:
                print(e)
                continue
            
            #读取WSI信息,设置可视化背景
            kfb_height = KFB_file.getHeight()
            kfb_width = KFB_file.getWidth()
            visulize_scale = 5 #可视化倍率
            visulize_height = int(kfb_height/(read_scale/visulize_scale))
            visulize_width = int(kfb_width/(read_scale/visulize_scale))
            wsi_background = KFB_file.ReadRoi(0, 0, visulize_width, visulize_height, visulize_scale)

            gray_img_list = [] #储存所有分类灰度热力图       
            patches_dataset = wsi_to_patches(KFB_file, patch_size, read_scale, kfb_height, kfb_width)
            patches_dataloader = DataLoader(patches_dataset, batch_size=20, shuffle=False, num_workers=10, drop_last=False)
            print('开始分析：{}'.format(kfb_name))
            for patches in tqdm(patches_dataloader):
                patches = [patch.numpy() for patch in patches] 
                results = inference_model(model, patches)
                results = [result.seg_logits.data.cpu().numpy().squeeze() for result in results]#patch_size结果图
                #将结果图转换为灰度热力图
                gray_imgs = [tensor_to_heatmaps(result) for result in results]
                gray_img_list.extend(gray_imgs)
            
            all_heat_map_list = [] #单通道图片存储列表
            index = 0
            # 将结果拼接回完整的WSI
            all_WSI = np.zeros((4, kfb_height, kfb_width), dtype=np.float32)
            for y in tqdm(range(0, kfb_height-patch_size, patch_size//2)):
                for x in range(0, kfb_width-patch_size, patch_size//2):
                    patches = gray_img_list[index][2:]#处理所有需要保留的分类
                    patches = patches*kernel
                    for i in range(4):
                        all_WSI[i, y:y+patch_size, x:x+patch_size] += patches[i]
                    index += 1
            # 调整结果大小并添加到 all_heat_map_list
            all_heat_map_list = [
                cv2.resize(all_WSI[i], (visulize_width, visulize_height), interpolation=cv2.INTER_NEAREST)
                for i in range(4)
            ]
            
            #合并倒数第三和第二层结果（合并粘液腺瘤低度和高度）
            combine_result = np.add(all_heat_map_list[-3], all_heat_map_list[-2])
            all_heat_map_list.append(combine_result)

            #保留第一，四， 五层结果, (阴性，细支气管腺瘤，粘液腺癌)
            wsi_with_heat_list = []
            none_zero_area_list = []
            for useful_layer in [0, 3, 4]:
                heat_map = all_heat_map_list[useful_layer]
                heat_map = (heat_map - 128)*2
                heat_map[heat_map < 0] = 0
                heat_map = heat_map.astype(np.uint8)
                area = np.count_nonzero(heat_map)#计算热力面积
                none_zero_area_list.append(area)
                heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_TURBO)#转换为彩色热力图
                wsi_with_heat = cv2.addWeighted(wsi_background, 0.5, heat_map, 0.5, 0)
                wsi_with_heat_list.append(wsi_with_heat)
                
            # 计算各类面积占比    
            proportions = [round(area / sum(none_zero_area_list), 4) for area in none_zero_area_list]
            # 判读切片的病变分类
            if proportions[1] > proportions[2]:
                wsi_diagnosis = '细支气管腺瘤'
            else:
                wsi_diagnosis = '粘液腺癌'

            all_heat_map = np.hstack(wsi_with_heat_list)
            cv2.imwrite(os.path.join(save_path, '{}.png'.format(kfb_name)), all_heat_map)
            with open(os.path.join(save_path, 'wsi_diagnosis.csv'), mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([kfb_name, kfb_path, wsi_diagnosis, proportions])
        except Exception as e:
            print(e)
            with open(os.path.join(save_path, 'wsi_diagnosis.csv'), mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([kfb_name, kfb_path, e, 'error'])
            


if __name__ == '__main__':
    kfb_csv = 'data/test_wsi/svs.csv' # 需要推理的kfb文件路径列表
    read_scale = 10 # 读取WSI的倍率
    patch_size = 512
    wsi_type = 'svs'

    config_path = 'segformer_b3_10x.py' # 模型配置文件路径
    checkpoint_path = 'work_dirs/segformer_b3_10x/iter_229689.pth' # 模型权重路径，权重文件不进入 Git
    save_path = 'data/test_wsi' # 结果保存路径

    run(kfb_csv, read_scale, patch_size, config_path, checkpoint_path, save_path, wsi_type)
