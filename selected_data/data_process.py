import shutil
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd
import datetime
import copy
import torch

class SelectedDataset(Dataset):
    def __init__(self, data_dir=None, name=None):
        self.name = name
        if data_dir is None:
            self.field = None
            self.label = None
            self.label = None
            self.field_dims = None
            return
        if self.name=="selected":
            data = pd.read_csv(data_dir, header=None).to_numpy()
            # np.random.shuffle(data)
            self.field = data[:, :-1].astype(np.int)
            self.label = data[:, -1].astype(np.int)
            self.field_dims = np.max(self.field, axis=0) + 1
            return
        if self.name=="amazon":
            data = pd.read_csv(data_dir, header=0, names=["user_id", "item_id", "brand_id", "rating"],
                               delimiter='\t', usecols=["user_id", "item_id", "brand_id", "rating"]).to_numpy()
            # np.random.shuffle(data)
            self.field = data[:, :-1].astype(np.int)
            self.label = data[:, -1].astype(np.float)
            self.label = np.where(self.label >= 4, 1, 0).astype(np.int)
            self.field_dims = np.max(self.field, axis=0) + 1
            return
        if self.name=="ml25m":
            data = pd.read_csv(data_dir, header=0, names=["user_id", "item_id", "title", "year", "rating"],
                               delimiter='\t', usecols=["user_id", "item_id", "year", "rating"]).to_numpy()
            # np.random.shuffle(data)
            self.field = data[:, :-1].astype(np.int)
            self.label = data[:, -1].astype(np.float)
            self.label = np.where(self.label >= 4, 1, 0).astype(np.int)
            self.field_dims = np.max(self.field, axis=0) + 1
            return
        raise ValueError('unknown dataset name: ' + name)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        field = self.field[item]
        label = self.label[item]
        return field, label

    def split(self, part=[0.8, 0.1]):
        if self.field is not None and self .label is not None:
            len_set = len(self.label)
            part = [int(p * len_set) for p in part]
            part[1]  += part[0]
            train_set = SelectedDataset()
            val_set = SelectedDataset()
            test_set = SelectedDataset()
            if len(self.field.shape)==2:
                train_set.field = self.field[:part[0], :]
                val_set.field = self.field[part[0]:part[1], :]
                test_set.field = self.field[part[1]:, :]
            else:
                train_set.field = self.field[:part[0]]
                val_set.field = self.field[part[0]:part[1]]
                test_set.field = self.field[part[1]:]

            train_set.label = self.label[:part[0]]
            val_set.label = self.label[part[0]:part[1]]
            test_set.label = self.label[part[1]:]

            train_set.field_dims = np.max(train_set.field, axis=0) + 1
            val_set.field_dims = np.max(val_set.field, axis=0) + 1
            test_set.field_dims = np.max(test_set.field, axis=0) + 1

            return train_set, val_set, test_set
        else:
            raise ValueError('no interaction in the dataset.')

class CrossDataset(Dataset):
    def __init__(self, target, source=None):
        if source is None:
            c = copy.deepcopy(target)
            self.overlap_user = None
            self.field = c.field
            self.label = c.label
            self.cross = torch.tensor(np.ones(c.field.shape[0]))
        else:
            c = copy.deepcopy(target)
            d = copy.deepcopy(source)
            self.overlap_user = torch.tensor(np.intersect1d(d.field[:,0], c.field[:,0]))
            self.field = np.vstack((d.field, c.field))
            self.label = np.hstack((d.label, c.label))
            # target=1, source=0
            self.cross = torch.tensor(np.hstack((np.zeros(d.field.shape[0]), np.ones(c.field.shape[0]))))
        self.field_dims = np.max(self.field, axis=0) + 1
    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        field = self.field[item]
        label = self.label[item]
        cross = self.cross[item]
        return field, label, cross

def MergeDataset(a,b):
    c = copy.deepcopy(b)
    d = copy.deepcopy(a)
    c.field = np.vstack((d.field, c.field))
    c.label = np.hstack((d.label, c.label))
    c.field_dims = np.max(c.field, axis=0) + 1
    return c

def CutDataset(a, threshold):
    b = copy.deepcopy(a)
    c = copy.deepcopy(a)
    cutlen = int(len(a)*threshold)
    b.field = b.field[:cutlen,:]
    b.label = b.label[:cutlen]
    b.field_dims = np.max(b.field, axis=0) + 1
    c.field = c.field[cutlen:, :]
    c.label = c.label[cutlen:]
    c.field_dims = np.max(c.field, axis=0) + 1
    return b,c



def save_selected_data(save_path, selected_fields, selected_target):
    slct_fields = copy.deepcopy(selected_fields)
    slct_target = copy.deepcopy(selected_target)
    new_array = np.hstack((slct_fields,slct_target[:,None]))
    pd.DataFrame(new_array).to_csv(save_path, mode='a', header=None, index = None)
    del new_array, slct_fields, slct_target

def clear_selected_data(save_path):
    pd.DataFrame(data=None).to_csv(save_path, mode='w', header=None, index = None)

def backup_best_data(save_path):
    shutil.copy(save_path, save_path.replace('train', 'best'))

def record_excel(record_path, data_array, head1, head2):
    try:
        writer = pd.ExcelWriter(record_path, mode="a", engine="openpyxl")
    except:
        writer = pd.ExcelWriter(record_path, engine="openpyxl")
    data = pd.DataFrame(data_array)
    sheet_name = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    data.to_excel(writer, sheet_name, float_format='%.8f', header=[head1, head2], index=False)
    writer.save()
    writer.close()
