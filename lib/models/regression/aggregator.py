import torch

from lib.models.regression.encoder.preact import PreActBlock


class CorrelationVolumeWarping(torch.nn.Module):
    def __init__(self, cfg, volume_channels):
        super().__init__()

        self.position_encoder = cfg.POSITION_ENCODER
        self.position_encoder_im1 = cfg.POSITION_ENCODER_IM1
        self.max_score_channel = cfg.MAX_SCORE_CHANNEL
        self.cv_out_layers = cfg.CV_OUTLAYERS
        self.cv_half_channels = cfg.CV_HALF_CHANNELS
        self.pos_encoder_channels = cfg.UPSAMPLE_POS_ENC
        self.dustbin = cfg.DUSTBIN
        self.normalise_dot_prod = cfg.NORMALISE_DOT

        self.num_out_layers = 2 * volume_channels
        self.num_out_layers += 2 if self.position_encoder else 0
        self.num_out_layers += 2 if self.position_encoder_im1 else 0
        self.num_out_layers += 1 if self.max_score_channel else 0

        if self.cv_out_layers > 0:
            self.CV_block = PreActBlock(4800, self.cv_out_layers)
            self.num_out_layers += self.cv_out_layers

        if self.pos_encoder_channels > 0:
            pos_encoder_input_channels = 0
            pos_encoder_input_channels += 2 if self.position_encoder else 0
            pos_encoder_input_channels += 2 if self.position_encoder_im1 else 0
            self.pos_encoder_block = PreActBlock(
                pos_encoder_input_channels, self.pos_encoder_channels)
            self.num_out_layers += self.pos_encoder_channels

        # create dustbin learnable parameters
        if self.dustbin:
            self.bin_score = torch.nn.parameter.Parameter(100*torch.ones(1, 1, 1))
            self.bin_feature = torch.nn.parameter.Parameter(
                torch.zeros(1, volume_channels, 1), requires_grad=False)

    def forward(self, vol0, vol1):
        assert vol0.shape == vol1.shape, 'Feature volumes shape must match'

        # reshape feature volumes
        B, D, H, W = vol0.shape
        vol0 = vol0.view(B, D, H * W)
        vol1 = vol1.view(B, D, H * W)

        # normalise features before dot product
        if self.normalise_dot_prod:
            vol0 = torch.nn.functional.normalize(vol0, dim=1)
            vol1 = torch.nn.functional.normalize(vol1, dim=1)

        # computes correlation volume
        # softmax along last dimension -> for each feature in vol0, gets a discrete distribution over vol1 features
        if self.cv_half_channels:
            cvolume = torch.bmm(vol0[:, :D//2].transpose(1, 2), vol1[:, :D//2])  # [B, H*W, H*W]
        else:
            cvolume = torch.bmm(vol0.transpose(1, 2), vol1)  # [B, H*W, H*W]

        # add learned bin score to correlation volume
        # add learned bin feature to vol1
        if self.dustbin:
            cvolume = torch.cat((cvolume, self.bin_score.repeat(B, 1, H*W)), dim=1)
            cvolume = torch.cat((cvolume, self.bin_score.repeat(B, H*W+1, 1)), dim=2)
            vol1 = torch.cat((vol1, self.bin_feature.repeat(B, 1, 1)), dim=2)

        # normalise correlation volume using softmax
        cvolume = torch.softmax(cvolume, dim=2)            # [B, H*W (+1), H*W (+1)]

        # warps vol1 using feature volume
        vol1w = torch.bmm(vol1, cvolume.transpose(1, 2))  # [B, D, H*W (+1)]
        if self.dustbin:
            vol1w = vol1w[:, :, :-1]                        # [B, D, H*W]

        cat_volumes = [vol0, vol1w]

        # adds u,v channels showing *average* position of the most prominent features
        if self.position_encoder:
            u = torch.linspace(-1, 1, H).to(vol0.device)
            v = torch.linspace(-1, 1, W).to(vol0.device)
            uu, vv = torch.meshgrid(u, v)
            grid = torch.stack([uu, vv], dim=0).view(2, H * W)  # [2, H*W]
            grid = grid.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, H*W]
            pos_encoder = torch.bmm(grid, cvolume[:, :H*W, :H*W].transpose(1, 2))  # [B, 2, H*W]
            cat_volumes.append(pos_encoder)
            if self.position_encoder_im1:
                cat_volumes.append(grid)

            # upsamples encoder features to multiple channels
            if self.pos_encoder_channels > 0:
                pos_encoder_features = torch.cat(
                    [pos_encoder, grid],
                    dim=1) if self.position_encoder_im1 else pos_encoder
                pos_encoder_features = pos_encoder_features.view(B, -1, H, W)
                pos_encoder_features = self.pos_encoder_block(pos_encoder_features)
                pos_encoder_features = pos_encoder_features.view(B, -1, H * W)
                cat_volumes.append(pos_encoder_features)

        # adds single channel showing *highest* score of a given feature vector in the other map
        # could help show the confidence in the matching (if max_score is low means multiple similar features)
        if self.max_score_channel:
            max_score = torch.max(cvolume, dim=2, keepdim=True)[
                0].transpose(1, 2)[..., :H*W]  # [B, 1, H*W]
            cat_volumes.append(max_score)

        # if cv_out_layers > 0, adds a 'reduced' correlation layer representation into the global volume
        if self.cv_out_layers > 0:
            cvolume_resized = cvolume[:, :H*W, :H*W].view(B, H*W, H, W)
            cvolume_reduced = self.CV_block(cvolume_resized)
            cvolume_reduced = cvolume_reduced.view(B, -1, H * W)
            cat_volumes.append(cvolume_reduced)

        agg_volume = torch.cat(cat_volumes, dim=1).view(B, -1, H, W)
        return agg_volume


class CorrelationVolumeWarpingQKV(torch.nn.Module):
    def __init__(self, cfg, volume_channels):
        super().__init__()

        self.position_encoder = cfg.POSITION_ENCODER
        self.max_score_channel = cfg.MAX_SCORE_CHANNEL
        self.normalise_dot_prod = cfg.NORMALISE_DOT
        self.residuals = cfg.RESIDUAL_ATT

        self.num_out_layers = 2 * volume_channels
        self.num_out_layers += 2 if self.position_encoder else 0
        self.num_out_layers += 1 if self.max_score_channel else 0

        self.Q_mlp = torch.nn.Conv2d(volume_channels, volume_channels, 1, bias=False)
        self.K_mlp = torch.nn.Conv2d(volume_channels, volume_channels, 1, bias=False)
        self.V_mlp = torch.nn.Conv2d(volume_channels, volume_channels, 1, bias=False)

    def forward(self, vol0, vol1):
        assert vol0.shape == vol1.shape, 'Feature volumes shape must match'

        # apply query, key, value MLPs
        q = self.Q_mlp(vol0)
        k = self.K_mlp(vol1)
        v0 = self.V_mlp(vol0)
        v1 = self.V_mlp(vol1)

        # add skip connection (residual) to Q, K, V vectors
        if self.residuals:
            q = q + vol0
            k = k + vol1
            v0 = v0 + vol0
            v1 = v1 + vol1

        # reshape volumes
        B, D, H, W = vol0.shape
        q = q.view(B, D, H * W)
        k = k.view(B, D, H * W)
        v0 = v0.view(B, D, H * W)
        v1 = v1.view(B, D, H * W)

        if self.normalise_dot_prod:
            q = torch.nn.functional.normalize(q, p=2., dim=1)
            k = torch.nn.functional.normalize(k, p=2., dim=1)

        # computes correlation volume
        # softmax along last dimension -> for each feature in q, gets a discrete distribution over k features
        cvolume = torch.bmm(q.transpose(1, 2), k)  # [B, H*W, H*W]
        #cvolume = torch.softmax(cvolume / (D ** 0.5), dim=2)
        cvolume = torch.softmax(cvolume, dim=2)

        # warps v1 using feature volume
        v1w = torch.bmm(v1, cvolume.transpose(1, 2))  # [B, D, H*W]

        cat_volumes = [v0, v1w]

        # adds u,v channels showing *average* position of the most prominent features
        if self.position_encoder:
            u = torch.linspace(-1, 1, H).to(vol0.device)
            v = torch.linspace(-1, 1, W).to(vol0.device)
            uu, vv = torch.meshgrid(u, v)
            grid = torch.stack([uu, vv], dim=0).view(2, H * W)  # [2, H*W]
            grid = grid.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, H*W]
            pos_encoder = torch.bmm(grid, cvolume.transpose(1, 2))  # [B, 2, H*W]
            cat_volumes.append(pos_encoder)

        # adds single channel showing *highest* score of a given feature vector in the other map
        # could help show the confidence in the matching (if max_score is low means multiple similar features)
        if self.max_score_channel:
            max_score = torch.max(cvolume, dim=2, keepdim=True)[0].transpose(1, 2)  # [B, 1, H*W]
            cat_volumes.append(max_score)

        agg_volume = torch.cat(cat_volumes, dim=1).view(B, -1, H, W)
        return agg_volume


class Concat(torch.nn.Module):
    def __init__(self, cfg, volume_channels):
        super().__init__()
        self.num_out_layers = 2 * volume_channels

    def forward(self, vol0, vol1):
        return torch.cat([vol0, vol1], dim=1)
