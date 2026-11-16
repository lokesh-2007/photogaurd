import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

class DeepDiffusionPhotoGuardEngine:
    def __init__(self, vae_model, unet_model=None, scheduler=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.vae = vae_model.to(device)
        self.unet = unet_model.to(device) if unet_model is not None else None
        self.scheduler = scheduler
        self.device = device
        self.vae.eval()
        if self.unet:
            self.unet.eval()
        
        for param in self.vae.parameters():
            param.requires_grad = False
        if self.unet:
            for param in self.unet.parameters():
                param.requires_grad = False

    def optimize_complex_perturbation(
        self, 
        image_tensor: torch.Tensor, 
        epsilon: float = 12.0 / 255.0, 
        steps: int = 50, 
        step_size: float = 2.5 / 255.0
    ) -> torch.Tensor:
        """
        Executes a high-strength PGD attack targeting both VAE latent space 
        and early UNet noise prediction divergence.
        """
        x_orig = image_tensor.clone().detach().to(self.device)
        # Initialize with random noise within the L-infinity epsilon ball
        x_adv = x_orig.clone().detach() + torch.FloatTensor(*x_orig.shape).uniform_(-epsilon, epsilon).to(self.device)
        x_adv.clamp_(0, 1)
        
        # Target representation (zeros vector to force complete collapse of content representation)
        with torch.no_grad():
            target_latent = torch.zeros_like(self.vae.encode(x_orig * 2.0 - 1.0).latent_dist.mean)

        momentum = torch.zeros_like(x_adv).to(self.device)
        mu = 1.0

        for step in range(steps):
            x_adv.requires_grad_()
            
            # Scale input for VAE [-1, 1]
            x_scaled = x_adv * 2.0 - 1.0
            latents = self.vae.encode(x_scaled).latent_dist.mean
            
            # Primary loss: Maximize distance from original representation (pushing toward zero-latent collapse)
            loss = F.mse_loss(latents, target_latent)
            
            # Secondary deep diffusion component if UNet is loaded
            if self.unet and self.scheduler:
                # Add dummy timestep noise for diffusion guidance destruction
                t = torch.randint(10, 500, (1,), device=self.device).long()
                noise = torch.randn_like(latents)
                noisy_latents = self.scheduler.add_noise(latents, noise, t)
                noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=torch.zeros((1, 77, 768), device=self.device)).sample
                loss += F.mse_loss(noise_pred, noise) * 0.5

            # Compute gradients
            grad = torch.autograd.grad(loss, [x_adv], retain_graph=False)[0]
            
            # Momentum update
            grad = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8) + mu * momentum
            momentum = grad

            # Gradient ascent step (maximizing structural corruption)
            x_adv = x_adv + step_size * torch.sign(grad)

            # Strict L-infinity projection and pixel clipping
            eta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x_orig + eta, min=0.0, max=1.0).detach()

        return x_adv

def protect_image_stream_v2(image_bytes: bytes, vae_model, unet_model=None, scheduler=None, epsilon_val=12.0/255.0) -> bytes:
    engine = DeepDiffusionPhotoGuardEngine(vae_model, unet_model, scheduler)
    
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.ToTensor()
    tensor_img = transform(pil_img).unsqueeze(0)

    protected_tensor = engine.optimize_complex_perturbation(tensor_img, epsilon=epsilon_val)

    protected_pil = transforms.ToPILImage()(protected_tensor.squeeze(0).cpu())
    
    output_buffer = io.BytesIO()
    protected_pil.save(output_buffer, format="PNG", compress_level=3)
    output_buffer.seek(0)
    
    return output_buffer.getvalue()