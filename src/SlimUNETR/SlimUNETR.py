import torch
import torch.nn as nn

from .Decoder import Decoder, SplitClassDecoder
from .Encoder import Encoder


class SlimUNETR(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        embed_dim=96,
        embedding_dim=64,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        r_up=(4, 2, 2, 2),
        dropout=0.3,
        use_split_classes=False,
    ):
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            embed_dim: deepest semantic channels
            embedding_dim: position code length
            channels: selection list of downsampling feature channel
            blocks: depth list of slim blocks
            heads: multiple set list of attention computations in parallel
            r: list of stride rate
            dropout: dropout rate
        Examples::
            # for 3D single channel input with size (128, 128, 128), 3-channel output.
            >>> net = SlimUNETR(in_channels=4, out_channels=3, embedding_dim=64)

            # for 3D single channel input with size (96, 96, 96), 2-channel output.
            >>> net = SlimUNETR(in_channels=1, out_channels=2, embedding_dim=27)

        """
        if use_split_classes:
            channels[0] //= 2

        super(SlimUNETR, self).__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            r_up=r_up
        )
        self.dropout = nn.Dropout(dropout)

        if use_split_classes:
            self.decoder = SplitClassDecoder(
                out_channels=out_channels,
                embed_dim=embed_dim,
                channels=channels,
                blocks=blocks,
                heads=heads,
                r=r,
                r_up=r_up
            )
        else:
            self.decoder = Decoder(
                out_channels=out_channels,
                embed_dim=embed_dim,
                channels=channels,
                blocks=blocks,
                heads=heads,
                r=r,
                r_up=r_up
            )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.hidden_states_out = list()

    def forward(self, x):

        embeding, self.hidden_states_out, data_shape = self.encoder(x)

        # print(x.shape)
        # for h in self.hidden_states_out:
        #     print(h.shape)
        # print(embeding.shape, data_shape)  # (B, C, W, H, Z)

        embeding = self.dropout(embeding)

        x = self.decoder(embeding, self.hidden_states_out, data_shape)
        self.hidden_states_out = [x] + self.hidden_states_out

        x = self.leaky_relu(x)
        return x


if __name__ == "__main__":
    x = torch.randn(size=(1, 1, 128, 128, 128))
    model = SlimUNETR(
        in_channels=1,
        out_channels=1,
        embed_dim=96,
        embedding_dim=64,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        dropout=0.3,
    )
    print(model(x).shape)
