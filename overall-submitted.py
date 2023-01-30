import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
import time
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.cmf_new import CMF
from torchfm.model.clfm_new import CLFM
from torchfm.model.dtcdr_new import DTCDR
from torchfm.network import kmax_pooling
from selected_data.data_process_new import CrossDataset, SelectedDataset, SelectedDataset25, SelectedDatasetAmazom, save_selected_data, clear_selected_data


class ControllerNetwork_instance(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.controller_losses = None
    
    def forward(self, x):
        embed_x = self.embedding(x)
        output_layer = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return output_layer

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 2))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

def set_random_seed(seed):
    print("* random_seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_dataset(name, path, Amazon=0):
    if name == 'selected_data':
        if Amazon>0:
            return SelectedDatasetAmazom(path)
        return SelectedDataset25(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, field_dims):
    if name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(32,32,32), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(32,32,32), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(32,32,32), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'cmf':
        return CMF(field_dims, embed_dim=16, alpha=0.5, lamda=0.0, gamma=0.0)
    elif name == 'clfm':
        return CLFM(field_dims, user_embedding_size=32, item_embedding_size=16, share_embedding_size=8, alpha=0.5,
                    reg_weight=1e-4)
    elif name == 'dtcdr':
        return DTCDR(field_dims, embedding_size=16, mlp_hidden_size=[16, 16], dropout_prob=0.3, alpha=0.3,
                     base_model="NeuMF")
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train_pre(model, optimizer, data_loader, criterion, device,usecross=False):
    model.train()
    if usecross:
        for i, (fields, label, cross) in enumerate(tqdm(data_loader)):
            has_target = True
            has_source = True
            fields, label = fields.long(), label.long()
            fields, label, cross = fields.to(device), label.to(device), cross.to(device)

            label = label.reshape((-1, 1))
            target_idx = torch.nonzero(cross).squeeze()
            if not(target_idx is None or target_idx.dim() == 0 or
                   target_idx.shape[0]== 0):
                target_idx = target_idx.reshape((-1, 1))
                target_fields = torch.gather(
                    fields, 0, target_idx.repeat(1, fields.shape[1]))
                target_label = (torch.gather(
                    label, 0, target_idx.repeat(1, label.shape[1]))).squeeze()
            else:
                has_target = False

            source_idx = torch.nonzero(1 - cross).squeeze()
            if not(source_idx is None or source_idx.dim() == 0 or
                   source_idx.shape[0]== 0):
                source_idx = source_idx.reshape((-1, 1))
                source_fields = torch.gather(
                    fields, 0, source_idx.repeat(1, fields.shape[1]))
                source_label = torch.gather(
                    label, 0, source_idx.repeat(1, label.shape[1])).squeeze()
            else:
                has_source = False


            if not has_target:
                y_sl = model(source_fields)
                loss_list_sl = criterion(y_sl, source_label.float())
                loss = loss_list_sl.mean()
            elif not has_source:
                y_target = model(target_fields)
                loss_list_target = criterion(y_target, target_label.float())
                loss = loss_list_target.mean()
            else:
                fields_all = torch.cat((target_fields, source_fields))
                label_all = torch.cat((target_label, source_label))
                y = model(fields_all)
                loss_list_all = criterion(y, label_all.float())
                loss = loss_list_all.mean()

            model.zero_grad()
            loss.backward()
            optimizer.step()
        return
    for i, (fields, target) in enumerate(tqdm(data_loader)):
        fields, target = fields.long(), target.long()
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss_list = criterion(y, target.float())
        loss = loss_list.mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()

def train_cdr_pre(model, optimizer, data_loader, device, overlap_user=None):
    model.train()
    if overlap_user is not None:
        for i, (fields, target, cross) in enumerate(tqdm(data_loader)):
            index = []
            noindex = []
            for j in range(fields.shape[0]):
                if fields[j,0] in overlap_user:
                    index.append(j)
                else:
                    noindex.append(j)
            if len(index)!=0:
                print(index)
                fields_overlap = fields[index]
                target_overlap = target[index]
                fields_overlap, target_overlap = fields_overlap.long().to(device), target_overlap.float().to(
                    device)
            else:
                fields_overlap = None
                target_overlap = None
            if len(noindex)!= 0:
                fields_no = fields[noindex]
                target_no = target[noindex]
                fields_no, target_no= fields_no.long().to(device), target_no.float().to(device)
            else:
                fields_no = None
                target_no = None
            cross = cross.long().to(device)
            loss = model.calculate_loss(fields_overlap, target_overlap, fields_no, target_no, cross)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        return
    for i, (fields, target, cross) in enumerate(data_loader):
        fields, target, cross = fields.long(), target.float(), cross.long()
        fields, target, cross = fields.to(device), target.to(device), cross.to(device)
        loss = model.calculate_loss(fields, target, cross)
        model.zero_grad()
        loss.backward()
        optimizer.step()


def test_pre(model, data_loader, device, overlap_user=None):
    model.eval()
    targets, predicts = [], []
    if overlap_user is not None:
        overlap_user = torch.tensor(overlap_user).long().to(device)
        with torch.no_grad():
            for fields, target in data_loader:
                index = []
                noindex = []
                for j in range(fields.shape[0]):
                    if fields[j, 0] in overlap_user:
                        index.append(j)
                    else:
                        noindex.append(j)
                if len(index) != 0:
                    fields_overlap = fields[index]
                    target_overlap = target[index]
                    fields_overlap, target_overlap = fields_overlap.long().to(device), target_overlap.float().to(
                        device)
                    y_o = model(fields_overlap, overlap=True)
                    targets.extend(target_overlap.tolist())
                    predicts.extend(y_o.tolist())
                if len(noindex) != 0:
                    fields_no = fields[noindex]
                    target_no = target[noindex]
                    fields_no, target_no = fields_no.long().to(device), target_no.float().to(device)
                    y_no = model(fields_no)
                    targets.extend(target_no.tolist())
                    predicts.extend(y_no.tolist())
        return roc_auc_score(targets, predicts), log_loss(targets, predicts)
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def test_a_batch(model, fields, target):
    model.eval()
    predicts = model(fields).tolist()
    targets = target.tolist()
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def train_noFullBatch(model_name, field_dims, learning_rate, weight_decay, valid_data_loader, controller, optimizer_controller, data_loader, criterion, device, epoch_i, ControllerLoss, epsilon):
    model = []
    optimizer = []
    mra = 1
    for x in range(mra):
        model.append(get_model(model_name, field_dims).to(device))
        optimizer.append(torch.optim.Adam(
            params=model[x].parameters(), lr=learning_rate, weight_decay=weight_decay))
    for i, (fields, label, cross) in enumerate(tqdm(data_loader)):
        fields, label, cross = fields.to(device), label.float().to(device), cross.to(device)
        for model_x in model:
            model_x.eval()
        controller.eval()
        has_target = True
        has_source = True
        with torch.no_grad():
            label = label.reshape((-1, 1))
            target_idx = torch.nonzero(cross).squeeze()
            if target_idx is None or target_idx.dim() == 0 or target_idx.shape[0]== 0:
                has_target = False
                print("Target is none")
            else:
                target_idx = target_idx.reshape((-1, 1))
                target_fields = torch.gather(
                    fields, 0, target_idx.repeat(1, fields.shape[1]))
                target_label = (torch.gather(
                    label, 0, target_idx.repeat(1, label.shape[1]))).squeeze()

            source_idx = torch.nonzero(1 - cross).squeeze()
            if source_idx is None or source_idx.dim() == 0 or source_idx.shape[0]== 0:
                has_source = False
                print("Source is none")
            else:
                source_idx = source_idx.reshape((-1, 1))
                source_fields = torch.gather(
                    fields, 0, source_idx.repeat(1, fields.shape[1]))
                source_label = torch.gather(
                    label, 0, source_idx.repeat(1, label.shape[1])).squeeze()

            auc = 0
            test_fields, test_label = next(iter(valid_data_loader))
            test_fields, test_label = test_fields.to(device), test_label.to(device)
            for x in range(mra):
                auc_x,_ = test_a_batch(model[x], test_fields, test_label)
                auc+=auc_x
            auc/=mra

        controller.train()
        output_layer = controller(source_fields)
        output_layer1 = output_layer.detach()
        with torch.no_grad():
            # threshold = torch.tensor(min(random.random(), 0.9)).to(device)
            threshold = torch.tensor(0.5).to(device)
            prob_instance = torch.softmax(output_layer1, dim=-1)

            sampled_actions = torch.where(prob_instance[:, 1] > threshold, 1, 0)
            sampled_actions = torch.tensor(
                [action if random.random() >= epsilon else -(action - 1) for action in sampled_actions]).to(device)
            prob_idx = torch.nonzero(sampled_actions).squeeze()
            if prob_idx is None or prob_idx.dim()==0 or prob_idx.shape[0]== 0:
                continue
            selected_label = torch.gather(source_label, 0, prob_idx)
            prob_idx = prob_idx.reshape((-1,1))
            selected_instance = torch.gather(
                source_fields, 0, prob_idx.repeat(1, source_fields.shape[1]))
            loss_list = None
            for x in range(mra):
                y = model[x](source_fields)
                loss_list = y.reshape((1,-1)).float() if loss_list is None else torch.cat((loss_list, y.reshape((1,-1)).float()))
            loss_list = torch.mean(loss_list,dim=0)

        for model_x in model:
            model_x.train()
        for x in range(mra):
            if not has_target:
                y_sl = model[x](selected_instance)
                loss_list_sl = criterion(y_sl, selected_label)
                loss = loss_list_sl.mean()
            elif not has_source:
                y_target = model[x](target_fields)
                loss_list_target = criterion(y_target, target_label)
                loss = loss_list_target.mean()
            else:
                fields = torch.cat((target_fields, selected_instance))
                label = torch.cat((target_label, selected_label))
                y = model[x](fields)
                loss_list_all = criterion(y, label)
                loss = loss_list_all.mean()

            model[x].zero_grad()
            loss.backward()
            optimizer[x].step()
            model[x].eval()

        if has_source:
            with torch.no_grad():
                auc1 = 0
                test_fields, test_label = test_fields.to(device), test_label.to(device)
                for x in range(mra):
                    auc_x, _ = test_a_batch(model[x], test_fields, test_label)
                    auc1 += auc_x
                auc1 /= mra

                loss_list1 = None
                for x in range(mra):
                    y = model[x](source_fields)
                    loss_list1 = y.reshape((1, -1)).float() if loss_list1 is None else torch.cat(
                        (loss_list1, y.reshape((1, -1)).float()))
                loss_list1 = torch.mean(loss_list1, dim=0)

                reward = (auc1-auc) * (loss_list - loss_list1)

            c_loss = torch.sum(ControllerLoss(output_layer, sampled_actions) * reward)

            controller.zero_grad()
            c_loss.backward()
            optimizer_controller.step()

    print("epoch: {}, Probability: {}".format(epoch_i, prob_instance[:5, 1]))



def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def train_re(field_dims,train_data_loader, valid_data_loader, test_data_loader,
             model_name, learning_rate, criterion, weight_decay, device, save_model_name, usecross=False):
    model = get_model(model_name, field_dims).to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(
        num_trials=3, save_path=save_model_name.replace('controller', 'model')+'.pt')
    train_start_time = time.time()
    for epoch_i in range(25):
        train_pre(model, optimizer, train_data_loader,
                  criterion, device, usecross=usecross)
        auc, logloss = test_pre(model, valid_data_loader, device)
        print('\tretrain epoch:', epoch_i,
              'validation: auc:', auc, 'logloss:', logloss)
        if not early_stopper.is_continuable(model, auc):
            print(f'\tvalidation: best auc: {early_stopper.best_accuracy}')
            break
    train_end_time = time.time()
    print("\tTime of retrain: {:.2f}min".format(
        (train_end_time - train_start_time)/60))
    model.load_state_dict(torch.load(save_model_name.replace('controller', 'model') + '.pt'))
    auc, logloss = test_pre(model, test_data_loader, device)
    print(f'\tretrain test auc: {auc}, logloss: {logloss}\n')
    return auc, logloss, model

def train_cdr(field_dims, train_data_loader, valid_data_loader, test_data_loader,
             model_name, learning_rate, weight_decay, device, save_model_name, epoch, overlap_user=None):
    model = get_model(model_name, field_dims).to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(
        num_trials=3, save_path=save_model_name)
    train_start_time = time.time()
    for epoch_i in range(epoch):
        train_cdr_pre(model, optimizer, train_data_loader, device, overlap_user)
        auc, logloss = test_pre(model, valid_data_loader, device, overlap_user)
        print('\tepoch:', epoch_i,
              'validation: auc:', auc, 'logloss:', logloss)
        if not early_stopper.is_continuable(model, auc):
            print(f'\tvalidation: best auc: {early_stopper.best_accuracy}')
            break
    train_end_time = time.time()
    print("\tTime of retrain: {:.2f}min".format(
        (train_end_time - train_start_time)/60))
    model.load_state_dict(torch.load(save_model_name))
    auc, logloss = test_pre(model, test_data_loader, device, overlap_user)
    print(f'\ttest auc: {auc}, logloss: {logloss}\n')
    return auc, logloss, model


def save_test_validset(data_loader, selected_data_path):
    clear_selected_data(selected_data_path)
    print('Start saving', selected_data_path)
    for (fields, target) in data_loader:
        save_selected_data(selected_data_path, fields.cpu(
        ).numpy().copy(), target.cpu().numpy().copy())
    print('Finish saving.')

def select_instance(data_set, selected_data_path, controller, device, select_ratio):
    clear_selected_data(selected_data_path)
    slct_number = int(select_ratio*len(data_set))
    fields, target = torch.from_numpy(data_set.field).to(device), torch.from_numpy(data_set.label).to(device)
    controller.eval()
    output_layer = controller(fields)
    prob_instance = torch.softmax(output_layer, dim=-1)
    prob_idx = kmax_pooling(prob_instance[:,1], 0, slct_number)
    if prob_idx is not None and prob_idx.dim != 0:
        if prob_idx.size(dim=0) != 0:
            print("Probability: ", prob_instance[prob_idx[:5], 1], prob_instance[prob_idx[-5:], 1])
            selected_target = torch.gather(target, 0, prob_idx)
            selected_instance = torch.gather(
                fields, 0, prob_idx.unsqueeze(1).repeat(1, fields.shape[1]))
            save_selected_data(selected_data_path, selected_instance.cpu().numpy(), selected_target.cpu().numpy())
        else:
            print("No sample selected!")
    else:
        print("No sample selected!")
    print("---------------------------------Select_ra: ", select_ratio)
    print("---------------------------------Select: ", len(selected_target))
    return slct_number

def train_by_select(dataset_trans, selected_data_path, controller, device,
                    trans_num, dataset_train, batch_size, field_dims, valid_data_loader,
                    model_name, learning_rate, criterion, weight_decay, save_model_name,
                    test_data_loader, select_ratio):
    slct_number = select_instance(dataset_trans, selected_data_path, controller, device,
                                  select_ratio)
    print('\t# end selection...\t total number:', slct_number, '\n\tpath:', selected_data_path)
    if slct_number != 0:
        dataset_select = SelectedDataset(selected_data_path)
        print("Select number: [{} / {}]".format(slct_number, trans_num))
        dataset_merge = CrossDataset(dataset_train, dataset_select)
        selected_data_loader = DataLoader(dataset_merge, batch_size=batch_size, shuffle=True)
        retrain_auc, retrain_logloss, model = train_re(field_dims, selected_data_loader, valid_data_loader,
                                                       valid_data_loader,
                                                       model_name, learning_rate, criterion, weight_decay, device,
                                                       save_model_name, usecross=True)

        auc, logloss = test_pre(model, test_data_loader, device)
        return auc, logloss, retrain_auc, retrain_logloss

def main(dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         select_ratio,
         epsilon,
         amazon,
         ratio_num,
         select_start,
         select_end):
    if amazon>0:
        #0.9
        trans_name = 'Automotive'
        #0.17
        # trans_name = 'Toys_and_Games'
        dataset_name = 'Industrial_and_Scientific'
    else:
        trans_name = 'ml-25m_11'
        dataset_name = 'ml-25m_10'
    print('Transfer Dataset:', trans_name)
    print('Dataset:', dataset_name)
    print('Dataset path:', dataset_path)
    path_source = './data/' + trans_name + '/' + trans_name + '.inter'
    path_target = './data/' + dataset_name + '/' + dataset_name + '.inter'
    print("-source dataset:", path_source)
    print("-target dataset:", path_target)

    device = torch.device(device)
    dataset_trans = get_dataset('selected_data', path_source, amazon)
    dataset_target = get_dataset('selected_data', path_target, amazon)
    dataset_train, dataset_valid, dataset_test = dataset_target.split()

    if model_name in ["cmf", "clfm", "dtcdr"]:
        info = '{}_{}_{}_{}'.format(model_name, dataset_name, str(epoch), str(batch_size))
        print('Training batch size:', batch_size)
        save_model_name = './{}/model_whole_'.format(
            save_dir) + info + '.pt'
        print('Best model save path:', save_model_name)

        dataset_train = CrossDataset(dataset_train, dataset_trans)

        overlap_user = None
        if model_name == 'dtcdr':
            overlap_user = CrossDataset(dataset_target, dataset_trans).overlap_user
        field_dims = []
        for i in range(len(dataset_train.field_dims)):
            field_dims.append(max(dataset_train.field_dims[i], dataset_valid.field_dims[i],
                                  dataset_test.field_dims[i]))

        train_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        valid_data_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
        if model_name == 'dtcdr':
            train_cdr(field_dims, train_data_loader, valid_data_loader, test_data_loader,
                     model_name, learning_rate, weight_decay, device, save_model_name, epoch, overlap_user)
        else:
            train_cdr(field_dims, train_data_loader, valid_data_loader, test_data_loader,
                     model_name, learning_rate, weight_decay, device, save_model_name, epoch)
        return
    criterion = torch.nn.BCELoss(reduction='none')
    info = '{}_{}_{}_{}_{}'.format(model_name, dataset_name, str(epoch), str(batch_size), str(select_ratio))
    save_model_name = './{}/controller_whole_'.format(
        save_dir) + info
    # selected_data_path = './selected_data/notFixed_whole_{}_{}_{}_{}_{}_train.txt'.format(
    #     model_name, trans_name, dataset_name, epsilon, select_ratio*100)
    selected_data_path = './selected_data/notFixed_whole_{}_{}_{}_{}_train.txt'.format(
        model_name, trans_name, dataset_name, epsilon)
    print('Training batch size:', batch_size)
    print('Size of selected data in a training batch:', select_ratio)
    print('epsilon:', epsilon)

    dataset_cross = CrossDataset(dataset_train, dataset_trans)

    train_data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    cross_data_loader = DataLoader(dataset_cross, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    field_dims = []
    for i in range(len(dataset_train.field_dims)):
        field_dims.append(max(dataset_cross.field_dims[i], dataset_valid.field_dims[i],
                              dataset_test.field_dims[i]))


    controller = ControllerNetwork_instance(
        field_dims, embed_dim=16, mlp_dims=(64, 64), dropout=0.2).to(device)
    optimizer_controller = torch.optim.Adam(
        params=controller.parameters(), lr=learning_rate*0.0001, weight_decay=weight_decay)
    ControllerLoss = nn.CrossEntropyLoss(reduction='none')


    print('\n********************************************* train *********************************************\n')
    train_start_time = time.time()
    trans_num = len(dataset_trans)
    for epoch_i in range(epoch):
        print("\n******************************Epoch: {}*********************************".format(epoch_i))
        train_noFullBatch(model_name, field_dims, learning_rate, weight_decay, valid_data_loader, controller, optimizer_controller, cross_data_loader,
                              criterion, device, epoch_i, ControllerLoss, epsilon)


    print(
        '\n\t========================= retrain start ==========================\n')
    retrain_start_time = time.time()

    print('\t# start selection...')
    max_auc = 0.0
    max_logloss = 1000000.0
    test_auc = 0
    test_logloss = 0
    max_ratio = 0.0
    max_slct = 0
    auc_list = [0]
    logloss_list = [0]
    x_list = [0.0]
    for select_ratio in np.linspace(select_start, select_end, num=ratio_num):
        x_list.append(select_ratio)
        auc, logloss, retrain_auc, retrain_logloss = train_by_select(dataset_trans, selected_data_path, controller, device,
                    trans_num, dataset_train, batch_size, field_dims, valid_data_loader,
                    model_name, learning_rate, criterion, weight_decay, save_model_name,
                    test_data_loader, select_ratio)
        slct_number = select_instance(dataset_trans, selected_data_path, controller, device,
                                      select_ratio)
        auc_list.append(auc)
        logloss_list.append(logloss)

        if retrain_auc>max_auc:
            max_ratio = select_ratio
            max_auc=retrain_auc
            max_logloss = retrain_logloss
            max_slct = slct_number
            test_auc = auc
            test_logloss = logloss


    target_auc, target_logloss, _ = train_re(field_dims, train_data_loader, valid_data_loader,
                                             test_data_loader,
                                             model_name, learning_rate, criterion, weight_decay, device,
                                             save_model_name)
    auc_list[0] = target_auc
    logloss_list[0] = target_logloss
    merge_auc = auc_list[-1]
    merge_logloss = logloss_list[-1]
    print('\t========================= retrain end ===========================\n')
    print("--------------------------------------------------------------------------------------------------------")
    retrain_end_time = time.time()
    print("Time of training: {:.2f}min".format(
        (retrain_start_time - train_start_time) / 60))
    print("Time of retraining: {:.2f}min".format(
        (retrain_end_time - retrain_start_time) / 60))

    print("Target: auc_{:.8f}  logloss_{:.8f}".format(
        target_auc, target_logloss))

    print("Merge: auc_{:.8f}  logloss_{:.8f}".format(
        merge_auc, merge_logloss))

    if max_slct != 0:
        print("Selection: auc_{:.8f}  logloss_{:.8f}".format(
            test_auc, test_logloss))
        print("Best Ratio: {}, Slct: [{} / {}]".format(max_ratio, max_slct, trans_num))
    retrain_performance=[max_auc,max_logloss]
    with open('Record_data/%s_%s_%s_notFixed_whole.txt' % (model_name, trans_name, dataset_name), 'a') as the_file:
        the_file.write('\nModel:%s\nDataset:%s\ntrain Time:%.2f,train Epoches: %d, batch_size: %d, epsilon: %s\n retrain performance: %s\n'
                   % (model_name, dataset_name, (retrain_end_time - train_start_time)/60, epoch_i+1, batch_size, str(epsilon), str(retrain_performance)))
    print("Ratio x_axis: ", x_list)
    print("AUC axis: ", auc_list)
    print("Logloss axis: ", logloss_list)
    print("Selected data save to: ", selected_data_path)



if __name__ == '__main__':
    # set_random_seed(56789)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--amazon', type=int, default=1)
    parser.add_argument(
        '--dataset_path', default='')
    parser.add_argument('--model_name', default='ipnn')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--ratio_num', type=int, default=1)
    parser.add_argument('--select_start', type=float, default=0.05)
    parser.add_argument('--select_end', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument(
        '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='save_model')
    parser.add_argument('--repeat_experiments', type=int, default=1)
    parser.add_argument('--select_ratio',  type=float, default=0.75)
    parser.add_argument('--epsilon', type=float, default=0.1)
    args = parser.parse_args()


    for i in range(args.repeat_experiments):
        main(args.dataset_path,
             args.model_name,
             args.epoch,
             args.learning_rate,
             args.batch_size,
             args.weight_decay,
             args.device,
             args.save_dir,
             args.select_ratio,
             args.epsilon,
             args.amazon,
             args.ratio_num,
             args.select_start,
             args.select_end)
