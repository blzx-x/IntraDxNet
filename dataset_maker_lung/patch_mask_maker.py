import os
from linux import kfbReader
import cv2
import json
import glob
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFile
from skimage.filters import threshold_otsu
from scipy.signal import savgol_filter
import multiprocessing
from shapely.geometry import Polygon
from statistic import statistic, split_dataset, wsi_classes_statistic
import csv
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def smooth_line(base_line):
    window_size = 5
    order = 3
    smooth_line=[]
    
    x_list = [x[0] for x in base_line]
    y_list = [y[1] for y in base_line] 

    x_list = savgol_filter(x_list, window_size, order)
    y_list = savgol_filter(y_list, window_size, order) 

    for i in range(len(x_list)):
        smooth_line.append([x_list[i], y_list[i]])
    return smooth_line

def areas_in_each_mask(mask, read_scale):
    class_index, pixels = np.unique(mask, return_counts=True)
    area_per_piexl = (10/read_scale)**2 #每个像素点对应的面积
    area = pixels*area_per_piexl
    return class_index.tolist(), area.tolist()
    
        
def resize_img(img, patch_size):
    if img.shape[0] != patch_size or img.shape[1] != patch_size:
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
    return img

def get_regions(sub_classes, seg_regions, ratio_mask):
    tumor_areas = []  # 存储所有闭合曲线  
    vertices = []
    
    for sub_class, seg_region in zip(sub_classes, seg_regions):  # 遍历每个闭合曲线
        
        if sub_class != 'ROI':
            for idx in range(len(seg_region)):  # 遍历每个闭合曲线的每个顶点
                x, y = int(seg_region[idx]['x'] // ratio_mask), int(seg_region[idx]['y'] // ratio_mask)
                vertices.append([x, y])
        else:
            x1, y1 = int(seg_region[0]['x'] // ratio_mask), int(seg_region[0]['y'] // ratio_mask)
            x2, y2 = int(seg_region[1]['x'] // ratio_mask), int(seg_region[1]['y'] // ratio_mask)
            vertices=[[x1, y1],[x2, y1],[x2, y2],[x1, y2]]
            
        #对曲线进行平滑
        if len(vertices) > 4:
            vertices = smooth_line(vertices)
        area = np.array(vertices).astype(np.int32)
        vertices = []
        area = area.tolist()
        tumor_areas.append(area)
    return tumor_areas

def check_percentage(elements, counts, class_num_mapper):
    total = sum(counts)  # 计算所有元素的总数量
    index_0 = elements.index(0) if 0 in elements else None  # 找到背景的索引，如果存在
    index_1 = elements.index(class_num_mapper['未标记的前景区域']) if class_num_mapper['未标记的前景区域'] in elements else None  # 找到'未标记的前景区域'的索引，如果存在
    
    count_0 = counts[index_0] if index_0 is not None else 0  # 获取0的数量，如果0不存在则为0
    count_1 = counts[index_1] if index_1 is not None else 0  
    
    # 计算0和1的总数量占所有元素总数量的百分比
    percentage = (count_0 + count_1) / total * 100
    percentage_2 = count_1 / total * 100
    
    
    return percentage < 90 and percentage_2 == 0 

    
def run(process_list, json_file_list, whether_replace, replace_from, replace_to, read_scale, patch_size, stride, csv_file):
    
    #创建保存信息的空list
    write_rows = []
    
    '''确定多进程中每个进程读取index'''
    lenth = len(json_file_list)
    process_index, processes_num, lock = process_list
    if process_index == processes_num -1:
        start_index = lenth//processes_num * process_index
        end_index = lenth
    else:
        start_index = lenth//processes_num * process_index
        end_index = lenth//processes_num * (process_index+1)  
    indexs = tqdm(range(start_index, end_index))
    for index in indexs:
        indexs.set_description('Process_{}'.format(process_index))
        json_file = json_file_list[index]

        try:
            with open(json_file, encoding='gbk') as f:
                json_info = json.load(f)
        except:
            with open(json_file, encoding='utf-8') as f:
                json_info = json.load(f)
            
        kfb_MD5 = json_info['KFB_BIND']['KFB_MD5']
        KFB_path = json_info['KFB_BIND']['KFB_FILE_PATH']
        if whether_replace == 'True':
            KFB_path = KFB_path.replace(replace_from,replace_to)
        KFB_path = KFB_path.replace('\\','/')
        KFB_file_name = KFB_path.split('/')[-1]
        KFB_file = kfbReader.reader()
        KFB_file.ReadInfo(KFB_path, 0, True)
        
        #此倍率只用于mask
        read_scale_mask = 5
        base_scale = KFB_file.getReadScale()
        ratio_mask = base_scale/read_scale_mask
        
        #获取kfb尺寸
        kfb_height = KFB_file.getHeight()
        kfb_width = KFB_file.getWidth()
        #计算mask尺寸
        mask_height = int(kfb_height/ratio_mask)
        mask_width = int(kfb_width/ratio_mask)

        
        sub_classes = [] #标注名称
        anno_classes =[] #标注对应的像素值
        seg_regions = []
        
        for j in range(len(json_info['annotation'])):
            if 'sub_class' in json_info['annotation'][j]:
                sub_class_name = 'sub_class'
            elif 'subClass' in json_info['annotation'][j]:
                sub_class_name = 'subClass'
            else:
                sub_class_name = 'NULL'
            
            if sub_class_name != 'NULL':
                
                sub_class = json_info['annotation'][j][sub_class_name]

                if sub_class in class_num_mapper:
                    sub_classes.append(sub_class)
                    anno_classes.append(class_num_mapper[sub_class])
                    seg_regions.append(json_info['annotation'][j]["points"])
        
        #处理annotation为空
        if seg_regions == []:
            print('seg curve is null')
            continue
        
        #获取tumor_areas
        tumor_areas = get_regions(sub_classes, seg_regions, ratio_mask)
        #根据标注面积从大到小排序
        polygons = [Polygon(p) for p in tumor_areas]
        areas = [p.area for p in polygons]

        # 获取排序后的索引，np.argsort 返回的是从小到大的索引，我们使用 [::-1] 来反转顺序实现从大到小排序
        sorted_indices = np.argsort(areas)[::-1]
        try:
            #以5倍率读取kfb文件，准备前景分割
            img_RGB = KFB_file.ReadRoi(0, 0, mask_width, mask_height, read_scale_mask)
            img_RGB = np.uint8(img_RGB)
            img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
            copy = img_RGB#可视化WSI
        except:
            print('Read WSI error')
            continue
        
        #分别创建训练mask和可视化mask画布
        mask = np.full((mask_height, mask_width), class_num_mapper["未标记的前景区域"] ,dtype="uint8")
        visiable_mask_color = color_type[str(class_num_mapper["未标记的前景区域"])] #未标记的前景区域对应的可视化颜色
        visiable_mask = np.zeros_like(img_RGB, dtype=np.uint8)#可视化mask
        visiable_mask[:] = visiable_mask_color
        
        #按先后排序绘制mask
        for sorted_index in sorted_indices:
            cv2.fillPoly(mask, [np.array(tumor_areas[sorted_index]).astype(np.int32)] , int(anno_classes[sorted_index])) 
            cv2.fillPoly(visiable_mask, [np.array(tumor_areas[sorted_index]).astype(np.int32)], color_type[str(anno_classes[sorted_index])])

        RGB_min = 5 #RGB最低像素值
        Magnifi_RGB = 1.15 #大津二值化阈值倍率
        Magnifi_HSV = 0.5 #大津二值化阈值倍率
        
        background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])*Magnifi_RGB
        background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])*Magnifi_RGB
        background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])*Magnifi_RGB
        
        #将True为背景改为True为前景
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        
        tissue_HSV = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])*Magnifi_HSV
        
        min_R = img_RGB[:, :, 0] > RGB_min
        min_G = img_RGB[:, :, 1] > RGB_min
        min_B = img_RGB[:, :, 2] > RGB_min
        
        tissue_combin = tissue_HSV & tissue_RGB & min_R & min_G & min_B
        
        
        # print('tissue combin down')

        #对组织mask进行腐蚀膨胀优化信息，并进行填充
        kernel = np.ones((3, 3), np.uint8)
        dilated_tissue_combin = cv2.dilate(tissue_combin.astype('uint8'), kernel, iterations=5)
        eroded_dilated_tissue_combin = cv2.erode(dilated_tissue_combin, kernel, iterations=5)
        
        tissue_mask = eroded_dilated_tissue_combin
        
        #标注文件输出的mask
        mask = mask.astype(np.int32)
        
        #获取前景与标注的交集
        mask =  tissue_mask*255 & mask  #非可视化mask
        visiable_mask = np.stack((tissue_mask*255, tissue_mask*255, tissue_mask*255), axis=-1) & visiable_mask  #可视化mask
        
        
        visiable_wsi = cv2.addWeighted(copy, 0.6, visiable_mask, 0.4, 60)
        cv2.imwrite(os.path.join(save_dir,'WSI',KFB_file_name.split('.')[0]+'.jpg'),visiable_wsi)
        
        #用于切片的图片倍率
        ratio = base_scale/read_scale
        patch_save_dir = os.path.join(save_dir,'Patch','patches')
        mask_8_save_dir = os.path.join(save_dir,'Patch','masks_8')
        mask_24_save_dir = os.path.join(save_dir,'Patch','masks_24')
        mask_on_patch_save_dir = os.path.join(save_dir,'Patch','mask_on_patch')
        
        # print('{}x Mask and WSI size: {} {}'.format(read_scale, int(kfb_width/ratio), int(kfb_height/ratio)))
        mask_for_patch = cv2.resize(mask, (int(kfb_width/ratio), int(kfb_height/ratio)), interpolation=cv2.INTER_NEAREST)
        mask_with_color = cv2.resize(visiable_mask, (int(kfb_width/ratio), int(kfb_height/ratio)), interpolation=cv2.INTER_NEAREST)
        #patch裁剪的左上角纵坐标
        start_patch_y = 0

        while start_patch_y < int(kfb_height/ratio) - patch_size:
            start_patch_x = 0 #patch裁剪的左上角横坐标
            #进度条
            
            while start_patch_x < int(kfb_width/ratio) - patch_size:
                
                cut_mask = mask_for_patch[start_patch_y:start_patch_y+patch_size,start_patch_x:start_patch_x+patch_size]
                cut_mask = resize_img(cut_mask, patch_size)
                class_index, area = areas_in_each_mask(cut_mask, read_scale)
                
                # 如果patch中背景面积大于90%，或者存在未标记前景区域，则跳过
                if check_percentage(class_index, area, class_num_mapper):
                    
                    #保存名称为图片序号加patch在当前倍率WSI下的左上角坐标
                    cv2.imwrite(os.path.join(mask_8_save_dir,'id{}_{}_{}.png'.format(index,start_patch_x,start_patch_y)), cut_mask)
                    
                    cut_patch = KFB_file.ReadRoi(start_patch_x, start_patch_y, patch_size, patch_size, read_scale)
                    cut_patch = np.uint8(cut_patch)
                    cut_patch = resize_img(cut_patch, patch_size)
                    cv2.imwrite(os.path.join(patch_save_dir,'id{}_{}_{}.png'.format(index,start_patch_x,start_patch_y)), cut_patch)
                    
                    cut_mask2 = mask_with_color[start_patch_y:start_patch_y+patch_size,start_patch_x:start_patch_x+patch_size]
                    cut_mask2 = resize_img(cut_mask2, patch_size)
                    mask_on_patch = cv2.addWeighted(cut_patch, 0.6, cut_mask2, 0.4, 60)
                    cv2.imwrite(os.path.join(mask_on_patch_save_dir,'id{}_{}_{}.png'.format(index,start_patch_x,start_patch_y)), mask_on_patch)
                    cv2.imwrite(os.path.join(mask_24_save_dir,'id{}_{}_{}.png'.format(index,start_patch_x,start_patch_y)), cut_mask2)
                    
                    write_rows.append([json_file, KFB_path, KFB_file_name, kfb_MD5, 
                                    os.path.join(patch_save_dir,'id{}_{}_{}.png'.format(index,start_patch_x,start_patch_y)), 
                                    os.path.join(mask_8_save_dir,'id{}_{}_{}.png'.format(index,start_patch_x,start_patch_y)),
                                    os.path.join(mask_24_save_dir,'id{}_{}_{}.png'.format(index,start_patch_x,start_patch_y)),
                                    'id{}_{}_{}.png'.format(index,start_patch_x,start_patch_y), class_index, area])
                                        
                start_patch_x += stride
            start_patch_y += stride
            
    with lock:
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(write_rows)
        
        
        
if __name__ == '__main__':
    
    current_path = os.path.dirname(__file__)
    config_file_path = os.path.join(current_path,'config/config.json')
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    json_dir = config['dirs']['json_dir']
    save_dir = config['dirs']['save_dir']
    class_num_mapper = config['class_label']
    read_scale = config['read_scale']
    patch_size = config['patch_size']
    stride = config['stride']
    whether_replace = config['replace_dir']['replace']
    replace_from = config['replace_dir']['replace_from']
    replace_to = config['replace_dir']['replace_to']
    color_type = config['color_type']
    random_seed = config["data_set_split"]['random_seed']
    train_valid_test = config["data_set_split"]['train_valid_test']
    json_file_list = glob.glob(os.path.join(json_dir,'*.json'))
    
    
    #在save_dir路径下创建WSI,Patch,Info文件夹
    if not os.path.exists(os.path.join(save_dir,'WSI')):
       os.makedirs(os.path.join(save_dir,'WSI'))
    if not os.path.exists(os.path.join(save_dir,'Patch')):
       os.makedirs(os.path.join(save_dir,'Patch'))
       os.makedirs(os.path.join(save_dir,'Patch','masks_8'))
       os.makedirs(os.path.join(save_dir,'Patch','masks_24'))
       os.makedirs(os.path.join(save_dir,'Patch','patches'))
       os.makedirs(os.path.join(save_dir,'Patch','mask_on_patch'))  
    if not os.path.exists(os.path.join(save_dir,'Info')):
       os.makedirs(os.path.join(save_dir,'Info'))
    
    shutil.copy(config_file_path, os.path.join(save_dir, 'Info', 'config.json'))
    #创建表格
    #创建csv表格
    csv_file = os.path.join(save_dir,'Info','data.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Json_path", "KFB_path", "KFB_name", "KFB_MD5","patch_path", "mask_8_path","mask_24_path","image_name","classes_in_mask","areas_in_mask(um**2)"])
  
    
    #设置多进程
    # processes_num = multiprocessing.cpu_count()
    processes_num = 6
    print(processes_num)
    lock = multiprocessing.Manager().Lock()
    

    for i in range(processes_num):
        if 'process_lists' not in locals():
            process_lists = [[i, processes_num, lock]]    
        else:
            process_lists = np.vstack((process_lists, (i, processes_num, lock)))
    
    processes = []

    for process_list in process_lists:
        p = multiprocessing.Process(target=run, args=(process_list, json_file_list, whether_replace, 
                                                      replace_from, replace_to, read_scale, patch_size, 
                                                      stride, csv_file))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    

    split_dataset(json_dir, save_dir, train_valid_test, random_seed)
    statistic(save_dir)
    wsi_classes_statistic(save_dir)
  