from .DehazeSNN import DehazeSNN


def build_S_model():
    model = DehazeSNN(in_chans=3, embed_dims=[24, 48, 96, 48, 24], depths=[2, 4, 8, 4, 2],
                      mlp_ratio=[4., 4., 4., 4., 4.], lif_bias=True, drop_path_rate=0.1,
                      patch_norm=True,
                      lif=4, lif_fix_tau=False, lif_fix_vth=False,
                      lif_init_tau=0.25, lif_init_vth=0.25)
    return model

def build_M_model():
    model = DehazeSNN(in_chans=3, embed_dims=[24, 48, 96, 48, 24], depths=[8, 12, 16, 12, 8],
                      mlp_ratio=[4., 4., 4., 4., 4.], lif_bias=True, drop_path_rate=0.1,
                      patch_norm=True,
                      lif=4, lif_fix_tau=False, lif_fix_vth=False,
                      lif_init_tau=0.25, lif_init_vth=0.25)
    return model

def build_L_model():
    model = DehazeSNN(in_chans=3, embed_dims=[24, 48, 96, 48, 24], depths=[8, 16, 32, 16, 8],
                      mlp_ratio=[4., 4., 4., 4., 4.], lif_bias=True, drop_path_rate=0.1,
                      patch_norm=True,
                      lif=4, lif_fix_tau=False, lif_fix_vth=False,
                      lif_init_tau=0.25, lif_init_vth=0.25)
    return model
