import torch
import torch.nn as nn
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, controller_mlp, kmax_pooling


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

class CLFM(torch.nn.Module):
    def __init__(self, field_dims, user_embedding_size, item_embedding_size, share_embedding_size, alpha, reg_weight):
        super(CLFM, self).__init__()

        # load parameters info
        self.user_embedding_size = user_embedding_size
        self.source_item_embedding_size = item_embedding_size
        self.target_item_embedding_size = item_embedding_size
        self.share_embedding_size = share_embedding_size
        self.alpha = alpha
        self.reg_weight = reg_weight
        assert 0 <= self.share_embedding_size <= self.source_item_embedding_size and \
               0 <= self.share_embedding_size <= self.target_item_embedding_size
        "The number of shared dimension must less than the dimension of both " \
        "the source item embedding and target item embedding"


        # define layers and loss
        self.source_user_embedding = nn.Embedding(field_dims[0], self.user_embedding_size)
        self.target_user_embedding = nn.Embedding(field_dims[0], self.user_embedding_size)

        self.source_item_embedding = nn.Embedding(field_dims[1], self.source_item_embedding_size)
        self.target_item_embedding = nn.Embedding(field_dims[1], self.target_item_embedding_size)
        if self.share_embedding_size > 0:
            self.shared_linear = nn.Linear(self.user_embedding_size, self.share_embedding_size, bias=False)
        if self.source_item_embedding_size - self.share_embedding_size > 0:
            self.source_only_linear = \
                nn.Linear(self.user_embedding_size, self.source_item_embedding_size - self.share_embedding_size,
                          bias=False)
        if self.target_item_embedding_size - self.share_embedding_size > 0:
            self.target_only_linear = \
                nn.Linear(self.user_embedding_size, self.target_item_embedding_size - self.share_embedding_size,
                          bias=False)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.source_reg_loss = EmbLoss()
        self.target_reg_loss = EmbLoss()

        # self.apply(xavier_normal_initialization)

    def forward_emb(self, user_e, item_e, is_target=True):
        factors = []
        if is_target:
            if self.share_embedding_size > 0:
                share_factors = self.shared_linear(user_e)
                factors.append(share_factors)
            if self.target_item_embedding_size - self.share_embedding_size > 0:
                only_factors = self.target_only_linear(user_e)
                factors.append(only_factors)
            factors = torch.cat(factors, dim=1)
            output = self.sigmoid(torch.mul(factors, item_e).sum(dim=1))
            return output

        if self.share_embedding_size > 0:
            share_factors = self.shared_linear(user_e)
            factors.append(share_factors)
        if self.source_item_embedding_size - self.share_embedding_size > 0:
            only_factors = self.source_only_linear(user_e)
            factors.append(only_factors)
        factors = torch.cat(factors, dim=1)
        output = self.sigmoid(torch.mul(factors, item_e).sum(dim=1))
        return output


    def forward(self, x):
        user_embedding = self.target_user_embedding(x[:,0])
        item_embedding = self.target_item_embedding(x[:,1])
        return self.forward_emb(user_embedding, item_embedding)

    def calculate_loss(self, x, label, cross):
        user = x[:, 0]
        item = x[:, 1]

        target_idx = torch.nonzero(cross).squeeze()
        if target_idx is None or target_idx.dim == 0:
            loss_t = 0.0
        else:
            target_user = self.target_user_embedding(torch.gather(
                user, 0, target_idx))
            target_item = self.target_item_embedding(torch.gather(
                item, 0, target_idx))
            target_label = torch.gather(
                label, 0, target_idx)
            p_target = self.forward_emb(target_user, target_item)
            loss_t = self.loss(p_target, target_label) + self.reg_weight * self.target_reg_loss(
            target_user,
            target_item)

        source_idx = torch.nonzero(1-cross).squeeze()
        if source_idx is None or source_idx.dim == 0:
            loss_s = 0.0
        else:
            source_user = self.source_user_embedding(torch.gather(
                user, 0, source_idx))
            source_item = self.source_item_embedding(torch.gather(
                item, 0, source_idx))
            source_label = torch.gather(
                label, 0, source_idx)
            p_source = self.forward_emb(source_user, source_item, is_target=False)
            loss_s = self.loss(p_source, source_label) + self.reg_weight * self.source_reg_loss(
            source_user,
            source_item)

        return loss_s * self.alpha + loss_t * (1 - self.alpha)
