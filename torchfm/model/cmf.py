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

class CMF(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, alpha=0.5, lamda=0.0, gamma=0.0):
        super(CMF, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_dim = embed_dim
        # self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        # self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.source_reg_loss = EmbLoss()
        self.target_reg_loss = EmbLoss()

    def forward_emb(self, user_e, item_e):
        return self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        user_e = embed_x[:, 0, :]
        item_e = embed_x[:, 1, :]

        return self.forward_emb(user_e, item_e)


    def calculate_loss(self, x, label, cross):
        # print(x.shape)
        # print(label.shape)
        # print(cross.shape)
        embed_x = self.embedding(x)
        user_e = embed_x[:, 0, :]
        item_e = embed_x[:, 1, :]

        target_idx = torch.nonzero(cross).squeeze()
        if target_idx is None or target_idx.dim == 0:
            loss_t = 0.0
        else:
            target_label = torch.gather(
                label, 0, target_idx)
            target_idx = target_idx.reshape((-1, 1))
            target_user = torch.gather(
                user_e, 0, target_idx.repeat(1, user_e.shape[1]))
            target_item = torch.gather(
                item_e, 0, target_idx.repeat(1, item_e.shape[1]))
            p_target = self.forward_emb(target_user, target_item)
            loss_t = self.loss(p_target, target_label) + \
                     self.gamma * self.target_reg_loss(target_user,
                                                       target_item)

        source_idx = torch.nonzero(1-cross).squeeze()
        if source_idx is None or source_idx.dim == 0:
            loss_s = 0.0
        else:
            source_label = torch.gather(
                label, 0, source_idx)
            source_idx = source_idx.reshape((-1, 1))
            source_user = torch.gather(
                user_e, 0, source_idx.repeat(1, user_e.shape[1]))
            source_item = torch.gather(
                item_e, 0, source_idx.repeat(1, item_e.shape[1]))
            p_source = self.forward_emb(source_user, source_item)
            loss_s = self.loss(p_source, source_label) + \
                     self.lamda * self.source_reg_loss(source_user,
                                                       source_item)

        return loss_s * self.alpha + loss_t * (1 - self.alpha)
