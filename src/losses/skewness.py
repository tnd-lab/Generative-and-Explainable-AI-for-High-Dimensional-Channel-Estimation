import numpy as np
import torch
from scipy import stats


class SkewCalculator:
    def __init__(self, device=None):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def __call__(self, tensor):
        """
        Compute skewness of a PyTorch tensor with flexible shape.
        If the tensor is multi-dimensional, computes skewness along the first dimension.

        :param tensor: PyTorch tensor of any shape
        :return: PyTorch tensor containing skewness value(s)
        """
        # Ensure input is a PyTorch tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32, device=self.device)

        # Move tensor to CPU and convert to numpy for scipy
        numpy_array = tensor.detach().cpu().numpy()

        # Compute skewness along the first dimension
        if numpy_array.ndim == 1:
            skewness = stats.skew(numpy_array, axis=0, bias=False)

            if np.isnan(skewness):
                skewness = 0.0
        else:
            skewness = stats.skew(numpy_array, axis=1, bias=False)

        # Convert back to PyTorch tensor and move to original device
        return torch.tensor(skewness, dtype=torch.float32, device=self.device)
