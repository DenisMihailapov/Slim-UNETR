import torch
import torch.nn as nn

from .Slim_UNETR_Block import Block


class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r):
        super(TransposedConvLayer, self).__init__()
        self.transposed = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.transposed(x)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        embed_dim=384,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        r_up=(4, 2, 2, 2)
    ):
        super(Decoder, self).__init__()
        self.SegHead = TransposedConvLayer(
            dim_in=channels[0], dim_out=out_channels, r=r_up[0]
        )
        self.TSconv3 = TransposedConvLayer(dim_in=channels[1], dim_out=channels[0], r=r_up[1])
        self.TSconv2 = TransposedConvLayer(dim_in=channels[2], dim_out=channels[1], r=r_up[2])
        self.TSconv1 = TransposedConvLayer(dim_in=embed_dim, dim_out=channels[2], r=r_up[3])

        block = []
        for _ in range(blocks[0]):
            block.append(Block(channels=channels[0], r=r[0], heads=heads[0]))
        self.block1 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[1]):
            block.append(Block(channels=channels[1], r=r[1], heads=heads[1]))
        self.block2 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)

    def forward(self, x, hidden_states_out, data_shape):

        x = x.reshape(*data_shape)
        x = self.block4(x)
        x = self.TSconv1(x)
        x = x + hidden_states_out[2]

        x = self.block3(x)
        x = self.TSconv2(x)
        x = x + hidden_states_out[1]

        x = self.block2(x)
        x = self.TSconv3(x)
        x = x + hidden_states_out[0]

        x = self.block1(x)

        x = self.SegHead(x)

        # print(x.shape)

        return x


class SplitClassDecoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        embed_dim=384,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        r_up=(4, 2, 2, 2)
    ):
        super(SplitClassDecoder, self).__init__()
        self.out_channels = out_channels
        self.SegHead_l = nn.ModuleList([])
        self.TSconv3_l = nn.ModuleList([])
        self.block2_l = nn.ModuleList([])
        self.block1_l = nn.ModuleList([])

        for _ in range(out_channels):
            self.SegHead_l.append(
                TransposedConvLayer(dim_in=channels[0], dim_out=1, r=r_up[0])
            )
            self.TSconv3_l.append(
                TransposedConvLayer(dim_in=channels[1], dim_out=channels[0], r=r_up[1])
            )

            block = []
            for _ in range(blocks[1]):
                block.append(Block(channels=channels[1], r=r[1], heads=heads[1]))
            self.block2_l.append(nn.Sequential(*block))

            block = []
            for _ in range(blocks[0]):
                block.append(Block(channels=channels[0], r=r[0], heads=heads[0]))
            self.block1_l.append(nn.Sequential(*block))

        # ----------------------------------------------------------------

        self.TSconv2 = TransposedConvLayer(dim_in=channels[2], dim_out=channels[1], r=r_up[2])
        self.TSconv1 = TransposedConvLayer(dim_in=embed_dim, dim_out=channels[2], r=r_up[3])

        block = []
        for _ in range(blocks[2]):
            block.append(Block(channels=channels[2], r=r[2], heads=heads[2]))
        self.block3 = nn.Sequential(*block)

        block = []
        for _ in range(blocks[3]):
            block.append(Block(channels=embed_dim, r=r[3], heads=heads[3]))
        self.block4 = nn.Sequential(*block)

    def forward(self, x, hidden_states_out, data_shape):

        x = x.reshape(*data_shape)
        x = self.block4(x)
        x = self.TSconv1(x)
        x = x + hidden_states_out[2]
        # print("block4", x.shape)

        x = self.block3(x)
        x = self.TSconv2(x)
        x = x + hidden_states_out[1]
        # print("block3", x.shape)

        x_out = []
        for i in range(self.out_channels):
            x_out.append(self.block2_l[i](x))
            x_out[-1] = self.TSconv3_l[i](x_out[-1])
            x_out[-1] = x_out[-1] + hidden_states_out[0]

            x_out[-1] = self.block1_l[i](x_out[-1])

            x_out[-1] = self.SegHead_l[i](x_out[-1])

        # print("blocks", len(x_out), x_out[-1].shape)

        x = torch.cat(x_out, dim=1)

        # print(x.shape)

        return x
