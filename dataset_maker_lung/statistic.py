from ast import literal_eval
import os
import json
import pandas as pd
import random



def statistic(save_dir):
    
    csv_file = os.path.join(save_dir,'Info','data.csv')
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except:
        df = pd.read_csv(csv_file, encoding='gbk')
    
    classes_in_mask = df['classes_in_mask'].tolist()
    areas_in_mask = df['areas_in_mask(um**2)'].tolist()
    dataset_type_list = df['dataset']
    
    writer = pd.ExcelWriter(os.path.join(save_dir,'Info','area_statistic.xlsx'), engine='xlsxwriter')
    
    for i in ['train', 'valid', 'test']:
        area_sums = {}
        for classes, areas, dataset_type in zip(classes_in_mask, areas_in_mask, dataset_type_list):
            if dataset_type == i:
                classes = literal_eval(classes)
                areas = literal_eval(areas)
                for class_, area in zip(classes, areas):
                    if class_ in area_sums:
                        area_sums[class_] += area
                    else:
                        area_sums[class_] = area
                
        classes = list(area_sums.keys())
        areas = list(area_sums.values())
        areas = [a/10**6 for a in areas]

        df = pd.DataFrame({'class': classes, 'area(mm^2)': areas}) 
        df.to_excel(writer, sheet_name=i, index=False)
    writer.save()
    
def split_dataset(json_dir, save_dir, train_valid_test, random_seed):
    filenames = os.listdir(json_dir)
    random.seed(random_seed)
    random.shuffle(filenames)
    
    total = sum(train_valid_test)
    # 将每个数字除以总和，得到归一化后的分布
    train_valid_test = [x / total for x in train_valid_test]
    
    # 确定每个数据集的大小
    total_files = len(filenames)
    num_train = round(total_files * train_valid_test[0])
    num_val = round(total_files * train_valid_test[1])

    
    # 划分数据集
    train_filenames = filenames[:num_train]
    val_filenames = filenames[num_train:num_train + num_val]
    test_filenames = filenames[num_train + num_val:]
    
    csv_file = os.path.join(save_dir,'Info','data.csv')
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except:
        df = pd.read_csv(csv_file, encoding='gbk')
    
    json_files = [filepath.split('/')[-1] for filepath in df['Json_path']]
    split_list = []
    for json_file in json_files:
        if json_file in train_filenames:
            split_list.append('train')
        elif json_file in val_filenames:
            split_list.append('valid')
        else:
            split_list.append('test')
    
    df['dataset'] = split_list
    df.to_csv(os.path.join(save_dir,'Info','data.csv'), encoding='utf-8', index=False)

def wsi_classes_statistic(save_dir):
    csv_file = os.path.join(save_dir,'Info','data.csv')
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except:
        df = pd.read_csv(csv_file, encoding='gbk')
    # 根据 'KFB_name' 列合并 'classes_in_mask' 列
    merged_data = df.groupby('KFB_name')['classes_in_mask'].apply(lambda x: list(set(sum(map(literal_eval, x), [])))).reset_index()
    merged_data.to_csv(os.path.join(save_dir,'Info','wsi_classes_statistic.csv'), index=False)
    
if __name__ == '__main__':
    
    current_path = os.path.dirname(__file__)
    config_file_path = os.path.join(current_path,'config/config.json')
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
        
    save_dir = config['dirs']['save_dir']
    json_dir = config['dirs']['json_dir']
    random_seed = config["data_set_split"]['random_seed']
    train_valid_test = config["data_set_split"]['train_valid_test']
    
    # split_dataset(json_dir, save_dir, train_valid_test, random_seed)
    statistic(save_dir)
    
    