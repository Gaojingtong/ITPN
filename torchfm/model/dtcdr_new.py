import torch
import torch.nn as nn
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, controller_mlp, kmax_pooling
import numpy as np
from recbole.model.layers import MLPLayers

class EmbLoss(nn.Module):

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
        assert self.base_model in ['NeuMF'], "based model {} is not supported! ".format(self.base_model)

        self.source_user_embedding = FeaturesEmbedding([field_dims[0]], self.embedding_size)
        self.source_item_embedding = FeaturesEmbedding(field_dims[1:], self.embedding_size)

        self.target_user_embedding = FeaturesEmbedding([field_dims[0]], self.embedding_size)
        self.target_item_embedding = FeaturesEmbedding(field_dims[1:], self.embedding_size)

        self.source_mlp_layers = MLPLayers([len(field_dims) * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        self.source_mlp_layers.logger = None
        self.source_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.target_mlp_layers = MLPLayers([len(field_dims) * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        self.target_mlp_layers.logger = None
        self.target_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.source_sigmoid = nn.Sigmoid()
        self.target_sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()

    def forward(self, x, overlap=False):
        output = self.neumf_forward(x[:, 0], x[:, 1:], overlap, 'target')
        return output

    def neumf_forward(self, user, item, overlap=False, domain='source'):
        user_source_e = self.source_user_embedding(user).reshape((user.shape[0], -1))
        user_target_e = self.target_user_embedding(user).reshape((user.shape[0], -1))
        item_source_e = self.source_item_embedding(item).reshape((item.shape[0], -1))
        item_target_e = self.target_item_embedding(item).reshape((item.shape[0], -1))
        # if overlap:
        #     user_e = torch.maximum(user_source_e, user_target_e)
        #     item_e = torch.maximum(item_source_e, item_target_e)
        # elif domain=='source':
        #     user_e = user_source_e
        #     item_e = item_source_e
        # else:
        #     user_e = user_target_e
        #     item_e = item_target_e
        user_e = torch.maximum(user_source_e, user_target_e)
        item_e = torch.maximum(item_source_e, item_target_e)
        if domain == 'source':
            output = self.source_sigmoid(self.source_predict_layer(self.source_mlp_layers(torch.cat((user_e, item_e), -1))))
        else:
            output = self.target_sigmoid(self.target_predict_layer(self.target_mlp_layers(torch.cat((user_e, item_e), -1))))
        return output.squeeze(-1)

    def calculate_loss(self, x_o, label_o, x_no, label_no, cross):
        loss_t_o = True
        loss_t_no = True
        loss_s_o = True
        loss_s_no = True
        if x_o is not None:
            user_o = x_o[:, 0]
            item_o = x_o[:, 1:]
            target_idx = torch.nonzero(cross).squeeze()
            if target_idx is None or target_idx.dim == 0:
                loss_t_o = False
            else:
                target_user_o = torch.gather(
                    user_o, 0, target_idx)
                print("target_o", target_user_o)
                target_item_o = torch.gather(
                    item_o, 0, target_idx.reshape((-1, 1)).repeat(1, item_o.shape[1]))
                target_label_o = torch.gather(
                    label_o, 0, target_idx)
                target_output_o = self.neumf_forward(target_user_o, target_item_o, overlap=True, domain='target')
                # loss_t = self.loss(target_output, target_label)

            source_idx = torch.nonzero(1-cross).squeeze()
            if source_idx is None or source_idx.dim == 0:
                loss_s_o = False
            else:
                source_user_o = torch.gather(
                    user_o, 0, source_idx)
                source_item_o = torch.gather(
                    item_o, 0, source_idx.reshape((-1, 1)).repeat(1, item_o.shape[1]))
                source_label_o = torch.gather(
                    label_o, 0, source_idx)
                source_output_o = self.neumf_forward(source_user_o, source_item_o, overlap=True, domain='source')
                # loss_s = self.loss(source_output, source_label)
        else:
            loss_t_o = False
            loss_s_o = False
        if x_no is not None:
            user_no = x_no[:, 0]
            item_no = x_no[:, 1:]
            target_idx = torch.nonzero(cross).squeeze()
            if target_idx is None or target_idx.dim == 0:
                loss_t_no = False
            else:
                target_user_no = torch.gather(
                    user_no, 0, target_idx)
                target_item_no = torch.gather(
                    item_no, 0, target_idx.reshape((-1, 1)).repeat(1, item_no.shape[1]))
                target_label_no = torch.gather(
                    label_no, 0, target_idx)
                target_output_no = self.neumf_forward(target_user_no, target_item_no, domain='target')
                # loss_t = self.loss(target_output, target_label)

            source_idx = torch.nonzero(1-cross).squeeze()
            if source_idx is None or source_idx.dim == 0:
                loss_s_no = False
            else:
                source_user_no = torch.gather(
                    user_no, 0, source_idx)
                source_item_no = torch.gather(
                    item_no, 0, source_idx.reshape((-1, 1)).repeat(1, item_no.shape[1]))
                source_label_no = torch.gather(
                    label_no, 0, source_idx)
                source_output_no = self.neumf_forward(source_user_no, source_item_no, domain='source')
        else:
            loss_t_no = False
            loss_s_no = False
        if loss_t_o:
            if loss_t_no:
                print("have overlap")
                target_output = torch.cat((target_output_o, target_output_no))
                target_label = torch.cat((target_label_o,target_label_no))
            else:
                target_output = target_output_o
                target_label = target_label_o
        else:
            target_output = target_output_no
            target_label = target_label_no
        if loss_s_o:
            if loss_s_no:
                print("have overlap")
                source_output = torch.cat((source_output_o, source_output_no))
                source_label = torch.cat((source_label_o,source_label_no))
            else:
                source_output = source_output_o
                source_label = source_label_o
        else:
            source_output = source_output_no
            source_label = source_label_no
        loss_t = self.loss(target_output, target_label)
        loss_s = self.loss(source_output,source_label)
        return loss_s * self.alpha + loss_t * (1 - self.alpha)

