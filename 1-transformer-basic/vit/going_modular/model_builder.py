import torch
from torch import nn

class MultiheadSelfAttensionBlock(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, num_heads: int = 12, attn_dropout: float = 0
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn  = nn.MultiheadAttention(embed_dim=embedding_dim,num_heads=num_heads,dropout=attn_dropout,batch_first=True)

    def forward(self,x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x, key=x, value=x, need_weights=False
        )

        return attn_output

class MLPBlock(nn.Module):
    

    def __init__(
        self, embedding_dim: int = 768, mlp_size: int = 3072, dropout: float = 0.1
    ) -> None:
        """
        _summary_

        Args:
            embedding_dim (int, optional): _description_. Defaults to 768.
            mlp_size (int, optional): _description_. Defaults to 3072.
            dropout (float, optional): _description_. Defaults to 0.1.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self,x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        mlp_dropout: float = 0.1,
        attn_drop_out: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = MLPBlock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )
        self.attn = MultiheadSelfAttensionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_drop_out
        )

    def forward(self,x):
        x = self.attn(x) + x
        x = self.mlp(x) + x
        return x


class PatchEmbedding(nn.Module):

    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embedding_dim: int = 768
    ) -> None:
        super().__init__()

        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.patch_size = patch_size

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0

        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)

class ViT(nn.Module):

    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embedding_dim: int = 768,
        mlp_size: int = 3072,
        num_heads: int = 12,
        attn_dropout: float = 0,
        mlp_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        assert image_size % patch_size ==0

        # embedding
        self.num_patches = (image_size * image_size) // (patch_size**2)
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True
        )
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True
        )
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim
        )

        # transformer decoder
        self.transformer_encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_size=patch_size,
                    mlp_dropout=mlp_dropout,
                    attn_drop_out=attn_dropout
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self,x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x_patched= self.patch_embedding(x)
        x_patched = torch.cat((class_token, x_patched), dim=1)
        x = self.position_embedding + x_patched
        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:,0])
        return x
