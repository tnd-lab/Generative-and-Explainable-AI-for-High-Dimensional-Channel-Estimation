import torch


# Compute the critic loss
def discriminator_loss(loss_func, real_pred, fake_pred):
    real_loss = loss_func(real_pred, torch.ones_like(real_pred))
    fake_loss = loss_func(fake_pred, torch.zeros_like(fake_pred))
    return real_loss + fake_loss


# Compute the generator loss
def generator_loss(loss_func, fake_pred):
    return loss_func(fake_pred, torch.ones_like(fake_pred))


# before compute the gradient penalty, we need compute the gradient of interpolated image
def get_gradient(critic, real, fake, epsilon, *args):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        critic: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = critic(mixed_images, *args)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


# then we compute the gradient penalty
def gradient_penalty_loss(critic, real, fake, *args):

    # get epsilon value
    epsilon = torch.rand(len(real), 1, 1, 1, device=real.device, requires_grad=True)

    # Compute the gradient
    gradient = get_gradient(critic, real, fake, epsilon, *args)

    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def l2_loss(fake, real):
    return torch.norm(fake - real, p=2)
