
from typing import List

import torch
from tqdm import tqdm

from gs.core.GaussianModel import GaussianModel
from gs.core.View import KnownView

import pandas as pd

from gs.helpers.loss import lpips_loss, psnr_loss, ssim_loss


def eval_views(cameras: List[KnownView], model: GaussianModel):
    """
    Evaluate the model against the cameras on PSNR, SSIM, and LPIPS.
    """
    with torch.no_grad():
        results = []
        for camera in tqdm(cameras):
            camera = camera.to(model.positions.device)
            predicted, _, _ = model.forward(camera)
            target = camera.image.cuda()
            results.append({
                "camera": camera.id,
                "psnr": psnr_loss(predicted, target).item(),
                "ssim": ssim_loss(predicted, target).item(),
                "lpips": lpips_loss(predicted, target).item()
            })
        return pd.DataFrame(results)
        

