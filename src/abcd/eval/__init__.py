from typing import List

import pandas as pd
import torch
from tqdm import tqdm

from abcd.core.GaussianModel import GaussianModel
from abcd.core.View import KnownView
from abcd.helpers.loss import lpips_loss, psnr_loss, ssim_loss


def eval_views(cameras: List[KnownView], model: GaussianModel):
    """
    Evaluate the model against the cameras on PSNR, SSIM, and LPIPS.
    """
    with torch.no_grad():
        results = []
        for camera in tqdm(cameras):
            camera.to(model.positions.device)
            predicted, _, _ = model.forward(camera)
            target = camera.image.to(model.positions.device)
            results.append(
                {
                    "camera": camera.id,
                    "psnr": psnr_loss(predicted, target).item(),
                    "ssim": ssim_loss(predicted, target).item(),
                    "lpips": lpips_loss(predicted, target).item(),
                }
            )
            camera.to("cpu")
        return pd.DataFrame(results)
