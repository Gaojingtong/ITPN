import torch

from torchfm.layer import FeaturesEmbedding, FeaturesLinear, AttentionalFactorizationMachine


class AttentionalFactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, attn_size, dropouts):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.afm = AttentionalFactorizationMachine(embed_dim, attn_size, dropouts)

    def forward(self, x):
        x = self.linear(x) + self.afm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))
