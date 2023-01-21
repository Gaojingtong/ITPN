import shutil
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd
import datetime
import copy

class SelectedDataset(Dataset):
    def __init__(self, data_dir):
        data = pd.read_csv(data_dir, header=None).to_numpy().astype(np.int)
        # np.random.shuffle(data)
        self.field = data[:,:-1]
        self.label = data[:,-1]
        self.field_dims = np.max(self.field, axis=0) + 1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        field = self.field[item]
        label = self.label[item]
        return field, label

class SelectedDataset25(Dataset):
    def __init__(self, data_dir=None):
        self.field = None
        self.label = None
        self.field_dims = None
        if data_dir is not None:
            data = pd.read_csv(data_dir, header=0, delimiter='\t').to_numpy().astype(np.float)[:, :3]
            # np.random.shuffle(data)
            self.field = data[:,:-1].astype(np.int)
            self.label = data[:,-1]
            self.label = np.where(self.label>=4, 1, 0).astype(np.int)
            self.field_dims = np.max(self.field, axis=0) + 1

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        field = self.field[item]
        label = self.label[item]
        return field, label

    def split(self, part=[0.8, 0.1]):
        len_set = len(self.label)
        part = [int(p * len_set) for p in part]
        part[1]  += part[0]
        train_set = SelectedDataset25()
        val_set = SelectedDataset25()
        test_set = SelectedDataset25()
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

def MergeDataset(a,b):
    c = copy.deepcopy(b)
    d = copy.deepcopy(a)
    c.field = np.vstack((d.field, c.field))
    c.label = np.hstack((d.label, c.label))
    # state = np.random.get_state()
    # np.random.shuffle(c.field)
    # np.random.set_state(state)
    # np.random.shuffle(c.label)
    # c.label = c.label.squeeze()
    c.field_dims = np.max(c.field, axis=0) + 1
    return c

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
