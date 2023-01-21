import torch
import torch.nn as nn
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, controller_mlp, kmax_pooling
import numpy as np
from recbole.model.layers import MLPLayers

class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

class DTCDR(torch.nn.Module):
    def __init__(self, field_dims, embedding_size, mlp_hidden_size, dropout_prob, alpha, base_model="NeuMF"):
        super(DTCDR, self).__init__()

        # load parameters info
        self.embedding_size = embedding_size
        self.mlp_hidden_size = mlp_hidden_size
        self.dropout_prob = dropout_prob
        self.base_model = base_model
        self.alpha = alpha
        assert self.base_model in ['NeuMF', 'DMF'], "based model {} is not supported! ".format(self.base_model)

        # define layers and loss
        if self.base_model == 'NeuMF':
            self.source_user_embedding = nn.Embedding(field_dims[0], self.embedding_size)
            self.source_item_embedding = nn.Embedding(field_dims[1], self.embedding_size)

            self.target_user_embedding = nn.Embedding(field_dims[0], self.embedding_size)
            self.target_item_embedding = nn.Embedding(field_dims[1], self.embedding_size)
            # with torch.no_grad():
            #     # 这里overlap，target没定义
            #     self.target_user_embedding.weight[self.target_num_users:].fill_(np.NINF)
            #     self.target_item_embedding.weight[self.target_num_items:].fill_(np.NINF)
            #     self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(np.NINF)
            #     self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(np.NINF)

            self.source_mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
            self.source_mlp_layers.logger = None  # remove logger to use torch.save()
            self.source_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

            self.target_mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
            self.target_mlp_layers.logger = None  # remove logger to use torch.save()
            self.target_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        # else:
        #     self.source_history_user_id, self.source_history_user_value, _ = dataset.history_user_matrix(
        #         domain='source')
        #     self.source_history_item_id, self.source_history_item_value, _ = dataset.history_item_matrix(
        #         domain='source')
        #     self.source_interaction_matrix = dataset.inter_matrix(form='csr', domain='source').astype(np.float32)
        #     self.source_history_user_id = self.source_history_user_id.to(self.device)
        #     self.source_history_user_value = self.source_history_user_value.to(self.device)
        #     self.source_history_item_id = self.source_history_item_id.to(self.device)
        #     self.source_history_item_value = self.source_history_item_value.to(self.device)
        #
        #     self.target_history_user_id, self.target_history_user_value, _ = dataset.history_user_matrix(
        #         domain='target')
        #     self.target_history_item_id, self.target_history_item_value, _ = dataset.history_item_matrix(
        #         domain='target')
        #     self.target_interaction_matrix = dataset.inter_matrix(form='csr', domain='target').astype(np.float32)
        #     self.target_history_user_id = self.target_history_user_id.to(self.device)
        #     self.target_history_user_value = self.target_history_user_value.to(self.device)
        #     self.target_history_item_id = self.target_history_item_id.to(self.device)
        #     self.target_history_item_value = self.target_history_item_value.to(self.device)
        #
        #     self.source_user_linear = nn.Linear(in_features=self.source_num_items, out_features=self.embedding_size,
        #                                         bias=False)
        #     self.source_item_linear = nn.Linear(in_features=self.source_num_users, out_features=self.embedding_size,
        #                                         bias=False)
        #     self.source_user_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)
        #     self.source_item_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)
        #
        #     self.target_user_linear = nn.Linear(in_features=self.target_num_items, out_features=self.embedding_size,
        #                                         bias=False)
        #     self.target_item_linear = nn.Linear(in_features=self.target_num_users, out_features=self.embedding_size,
        #                                         bias=False)
        #     self.target_user_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)
        #     self.target_item_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)

        self.source_sigmoid = nn.Sigmoid()
        self.target_sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()
        # self.apply(xavier_normal_initialization)

    def forward(self, x):
        if self.base_model == 'NeuMF':
            output = self.neumf_forward(x[:,0], x[:,1], 'target')
            return output
        else:
            output = self.dmf_forward(x[:,0], x[:,1], 'target')
            return output

    def neumf_forward(self, user, item, domain='source'):
        user_source_e = self.source_user_embedding(user)
        user_target_e = self.target_user_embedding(user)
        user_e = torch.maximum(user_source_e, user_target_e)

        item_source_e = self.source_item_embedding(item)
        item_target_e = self.target_item_embedding(item)
        item_e = torch.maximum(item_source_e, item_target_e)

        if domain == 'source':
            output = self.source_sigmoid(self.source_predict_layer(self.source_mlp_layers(torch.cat((user_e, item_e), -1))))
        else:
            output = self.target_sigmoid(self.target_predict_layer(self.target_mlp_layers(torch.cat((user_e, item_e), -1))))
        return output.squeeze(-1)

    def calculate_loss(self, x, label, cross):
        user = x[:, 0]
        item = x[:, 1]

        target_idx = torch.nonzero(cross).squeeze()
        if target_idx is None or target_idx.dim == 0:
            loss_t = 0.0
        else:
            target_user = torch.gather(
                user, 0, target_idx)
            target_item = torch.gather(
                item, 0, target_idx)
            target_label = torch.gather(
                label, 0, target_idx)
            if self.base_model == 'NeuMF':
                target_output = self.neumf_forward(target_user, target_item, 'target')
                loss_t = self.loss(target_output, target_label)
            else:
                target_output = self.dmf_forward(target_user, target_item, 'source')
                loss_t = self.loss(target_output, target_label)

        source_idx = torch.nonzero(1-cross).squeeze()
        if source_idx is None or source_idx.dim == 0:
            loss_s = 0.0
        else:
            source_user = torch.gather(
                user, 0, source_idx)
            source_item = torch.gather(
                item, 0, source_idx)
            source_label = torch.gather(
                label, 0, source_idx)
            if self.base_model == 'NeuMF':
                source_output = self.neumf_forward(source_user, source_item, 'source')
                loss_s = self.loss(source_output, source_label)
            else:
                source_output = self.dmf_forward(source_user, source_item, 'source')
                loss_s = self.loss(source_output, source_label)
        return loss_s * self.alpha + loss_t * (1 - self.alpha)

