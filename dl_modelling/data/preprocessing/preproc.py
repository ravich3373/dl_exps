import torch
import pandas as pd
import torchvision.transforms as T


def default_tsfm(nc=1, expand_channel=False, mean=[0,], std=[255]):
    if nc == 3 and len(mean) == 1:
        mean = mean * 3
        std = std * 3
    default_tsfm = T.Compose(
        [T.ToTensor(),
         T.Lambda(lambda x: x.float()),
         T.Lambda(lambda x: x.repeat(3, 1, 1) if nc==3 else x),
         T.Normalize(mean, std)]
    )
    return default_tsfm



