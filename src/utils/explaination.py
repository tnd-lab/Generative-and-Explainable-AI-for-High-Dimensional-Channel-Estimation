import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Hook:
    def __init__(self, module):
        self.stored = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # Store the forward activations
        self.stored = output.detach()


class HookBwd:
    def __init__(self, module):
        self.stored = None
        module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module, grad_input, grad_output):
        # Store the gradients during the backward pass
        self.stored = grad_output[0].detach()


class GradCam:
    def __init__(self, model, conv_layer):
        self.model = model

        self.hook_activation = Hook(conv_layer)
        self.hook_gradient = HookBwd(conv_layer)

    def compute_activation_map(self, image):

        output = self.model(image)

        # because in our case ciritic is a single value so it is not possible to get the target class
        output.backward()

        # Retrieve stored activations and gradients
        activations = self.hook_activation.stored
        gradients = self.hook_gradient.stored

        # get importance weights of the gradients
        importance_weights = gradients.mean(dim=[2, 3], keepdim=True)

        # Multiply the importance weights with the activations
        weighted_activations = (
            importance_weights * activations
        )  # Element-wise multiplication

        # Get activation map by summing the weighted activations
        activation_map = weighted_activations.sum(dim=1)

        # Apply ReLU to remove negative values
        activation_map = F.relu(activation_map)

        # normalize the activation map
        activation_map = activation_map.cpu().detach().numpy()  # Convert to numpy
        activation_map = (activation_map - activation_map.min()) / (
            activation_map.max() - activation_map.min()
        )  # Normalize

        return activation_map

    def visualize_cam(self, image, save_path=None):
        # Compute the activation map
        activation_map = self.compute_activation_map(image)

        # Get the first image and activation map
        activation_map = activation_map.squeeze(0)
        image = image.squeeze(0)

        # Normalize the image
        image = (
            image.cpu().numpy().transpose(1, 2, 0)
        )  # Shape: [height, width, channels]
        image = (image - image.min()) / (image.max() - image.min())

        real_image = image[:, :, 0]

        # Plot the images side by side
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))

        # Plot the original image
        axs[0].imshow(
            real_image,
            alpha=0.9,
        )
        axs[0].axis("off")
        axs[0].set_title("Input Image")

        # Plot heatmap
        im = axs[1].imshow(
            activation_map,
            alpha=0.4,
            cmap="jet",
            extent=(0, image.shape[1], image.shape[0], 0),
            interpolation="bilinear",
        )
        axs[1].axis("off")
        axs[1].set_title("Activation Map")

        # Plot the overlay image
        axs[2].imshow(
            real_image, aspect="auto", alpha=0.9
        )  # Set higher alpha for real_image
        axs[2].imshow(
            activation_map,
            alpha=0.6,
            cmap="jet",
            extent=(0, image.shape[1], image.shape[0], 0),
            interpolation="bilinear",
        )
        axs[2].axis("off")
        axs[2].set_title("Activation Map Overlay")

        # set color bar
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # save file
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            np.save(save_path.replace(".png", ".npy"), activation_map)

        # plt.show()
        plt.close()


class ScoreCam(GradCam):
    def __init__(self, model, conv_layer):
        super(ScoreCam, self).__init__(model, conv_layer)

    def compute_activation_map(self, image):
        # Forward pass to get activation maps
        _ = self.model(image)
        activations = self.hook_activation.stored

        # Upsample activations to match input size
        upsampled = F.interpolate(
            activations, size=image.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalize each activation map
        normalized = torch.zeros_like(upsampled)
        for i in range(upsampled.shape[1]):
            normalized[:, i, :, :] = (
                upsampled[:, i, :, :] - upsampled[:, i, :, :].min()
            ) / (upsampled[:, i, :, :].max() - upsampled[:, i, :, :].min() + 1e-8)

        # Compute scores for each activation map
        scores = []
        for i in range(normalized.shape[1]):
            masked_input = image * normalized[:, i, :, :].unsqueeze(1)
            output = self.model(masked_input)
            scores.append(output.detach().cpu().numpy())

        scores = np.array(scores)

        # Weight the activation maps by their scores
        weighted_activations = (
            torch.tensor(scores, device=activations.device).unsqueeze(0) * activations
        ).sum(dim=1)

        # Apply ReLU and normalize
        activation_map = F.relu(weighted_activations)
        activation_map = activation_map.cpu().detach().numpy()
        activation_map = (activation_map - activation_map.min()) / (
            activation_map.max() - activation_map.min() + 1e-8
        )

        return activation_map

    def visualize_cam(self, image, save_path=None):
        return super().visualize_cam(image, save_path)


class EigenCam(GradCam):
    def __init__(self, model, conv_layer):
        super(EigenCam, self).__init__(model, conv_layer)

    def compute_activation_map(self, image):
        """
        Computes EigenCAM activation map using the eigenvector from the activations' covariance matrix.
        """
        # Step 1: Forward pass to get activation maps
        _ = self.model(image)
        activations = (
            self.hook_activation.stored
        )  # [batch_size, channels, height, width]

        # Step 2: Reshape activations to 2D (batch, channels, width*height)
        activations = activations.view(
            activations.size(0), activations.size(1), -1
        )  # [batch_size, channels, width*height]

        # Step 3: Compute the covariance matrix for each activation map (batch-wise)
        # Using torch.bmm for batch-wise matrix multiplication of activations and their transpose
        cov_matrix = torch.bmm(
            activations, activations.transpose(1, 2)
        )  # [batch_size, channels, channels]

        # Step 4: Move to CPU for eigenvalue computation (or stay on GPU if desired)
        cov_matrix = cov_matrix.cpu().detach().numpy()

        # Step 5: Compute eigenvalues and eigenvectors (using NumPy as PyTorch doesn't have batch-wise eigenvalue decomposition)
        eigenvalues, eigenvectors = np.linalg.eigh(
            cov_matrix
        )  # Eigenvalue decomposition for each sample in batch

        # Step 6: Use the eigenvector corresponding to the largest eigenvalue
        # The last eigenvector corresponds to the largest eigenvalue because they are sorted in ascending order
        largest_eigenvector = eigenvectors[..., -1]  # [batch_size, channels]

        # Convert eigenvector back to PyTorch and move to same device as the input image
        largest_eigenvector = torch.from_numpy(largest_eigenvector).to(
            image.device
        )  # [batch_size, channels]
        largest_eigenvector = torch.abs(largest_eigenvector)  # Take the absolute value

        # Step 7: Weight the activations by the largest eigenvector
        # Unsqueeze to match the activation dimensions, then apply weighting
        weighted_activations = activations * largest_eigenvector.unsqueeze(
            -1
        )  # [batch_size, channels, width*height]

        # Step 8: Sum across channels to get a single activation map
        activation_map = weighted_activations.sum(dim=1)  # [batch_size, width*height]

        # Step 9: Reshape back to original spatial dimensions (height, width)
        activation_map = activation_map.view(
            activations.size(0),
            self.hook_activation.stored.size(2),
            self.hook_activation.stored.size(3),
        )  # [batch_size, height, width]

        # Step 10: Apply ReLU and normalize the activation map
        activation_map = F.relu(activation_map)
        activation_map = activation_map.cpu().detach().numpy()

        # Normalize the heatmap to be between 0 and 1
        activation_map = (activation_map - activation_map.min()) / (
            activation_map.max() - activation_map.min() + 1e-8
        )

        return activation_map

    def visualize_cam(self, image, save_path=None):
        return super().visualize_cam(image, save_path)
