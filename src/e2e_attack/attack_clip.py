"""
This code implements the "Camera-Robust" attack we discussed. Unlike standard attacks that work on static images, this uses Expectation over Transformation (EoT).
It simulates the printing and scanning process (rotations, noise, perspective changes) during the optimization loop so that the adversarial noise becomes robust
enough to survive a student taking a photo of the exam.

1. The Input: Replace dummy_exam.jpg with a scan of a real exam paper.

2. The Target: I set the target to "a photo of a toaster". When you run this, the resulting image will look like an exam paper to you,
   but if you feed it into a CLIP-based model (like LLaVA or older GPT-4V versions), it will classify the image as a toaster.

3. The "Invisible" Constraint: The variable EPSILON = 16/255 controls how much "gray static" is allowed on the page.
    (a) Higher Epsilon: Stronger attack, but the paper looks dirty/noisy to students.
    (b) Lower Epsilon: Harder to detect by human eye, but might fail if the camera angle is too extreme.

4. Run this script on a sample PDF page converted to an image. Then, take the output exam_adversarial.png, display it on your screen,
   and try to use Google Lens or a ChatGPT vision feature on your phone to scan it.
    If the model struggles to transcribe the text or identifies it as something random, you have successfully created a physical adversarial example.

"""

# pip install torch torchvision ftfy regex tqdm git+https://github.com/openai/CLIP.git
# pip install git+https://github.com/openai/CLIP.git

import torch
import clip
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"  # Standard CLIP model
LR = 0.01  # Learning rate for the noise
EPSILON = 64 / 255  # Max perturbation magnitude (approx 0.06)
STEPS = 500  # Optimization steps
EOT_SAMPLES = 5  # Number of random transforms per step (Robustness)


# ==========================================
# 2. EXPECTATION OVER TRANSFORMATION (EoT)
# ==========================================
class EoT_Transform(nn.Module):
    """
    Simulates the physical world process: Printing -> Camera -> Digital.
    This makes the attack robust to rotation, scaling, and noise.
    """

    def __init__(self, output_size=224):
        super().__init__()
        self.output_size = output_size
        self.aug = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
            transforms.RandomResizedCrop(output_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])

        # Standard CLIP normalization
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )

    def forward(self, x):
        # x is (1, C, H, W) high-res image
        # Apply random physical augmentations
        x_aug = self.aug(x)

        # Add random "camera noise" (Gaussian noise) to simulate sensor grain
        noise = torch.randn_like(x_aug) * 0.05
        x_aug = x_aug + noise

        # Clip to valid image range [0,1]
        x_aug = torch.clamp(x_aug, 0, 1)

        # Normalize for CLIP
        return self.normalize(x_aug)


# ==========================================
# 3. THE ATTACK LOGIC (PGD)
# ==========================================
def run_attack(image_path, target_text_str="a white piece of paper"):
    print(f"[Info] Loading CLIP model: {MODEL_NAME}...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()  # Freeze model weights

    # Load high-res original image
    # We convert to tensor manually to keep high resolution for the 'canvas'
    original_pil = Image.open(image_path).convert("RGB")
    high_res_transform = transforms.ToTensor()

    # x_clean is the high-res canvas (e.g., A4 size)
    x_clean = high_res_transform(original_pil).unsqueeze(0).to(DEVICE)

    # Initialize Perturbation (delta)
    # We optimize delta on the HIGH RES image, not the resized one.
    delta = torch.zeros_like(x_clean, requires_grad=True).to(DEVICE)

    # Setup EoT simulator
    eot = EoT_Transform(output_size=224).to(DEVICE)

    # Define Target: We want the image to look like 'target_text_str' to the model
    # OR we want to maximize distance from "exam paper".
    # Here we use a targeted attack: Force it to be seen as something irrelevant.
    text_inputs = clip.tokenize([target_text_str]).to(DEVICE)
    with torch.no_grad():
        target_embedding = model.encode_text(text_inputs)
        target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

    optimizer = torch.optim.Adam([delta], lr=LR)

    print(f"[Info] Starting PGD Attack with EoT...")
    for step in tqdm(range(STEPS)):
        optimizer.zero_grad()

        loss = 0
        # EoT Loop: Average gradients over multiple random "camera angles"
        for _ in range(EOT_SAMPLES):
            # 1. Apply perturbation to clean image
            # Clamp ensures the physical print is still valid RGB
            x_adv_high_res = torch.clamp(x_clean + delta, 0, 1)

            # 2. Simulate Camera/Scanning (Resize, Rotate, Noise)
            x_adv_model_input = eot(x_adv_high_res)

            # 3. Get Image Embedding
            image_features = model.encode_image(x_adv_model_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 4. Calculate Loss (Negative Cosine Similarity)
            # We want similarity to "toaster" (or whatever target) to be HIGH
            # So we minimize: 1 - CosineSim
            current_sim = torch.matmul(image_features, target_embedding.T)
            loss += (1 - current_sim)

        # Average loss over EoT samples
        loss = loss / EOT_SAMPLES

        # 5. Backprop
        loss.backward()

        # 6. Update Delta (PGD step)
        optimizer.step()

        # 7. Project Delta (Constraint to keep document readable)
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -EPSILON, EPSILON)

    # ==========================================
    # 4. SAVE RESULTS
    # ==========================================
    # Create final adversarial image
    x_final = torch.clamp(x_clean + delta, 0, 1)

    # Convert back to PIL
    adv_img_pil = transforms.ToPILImage()(x_final.squeeze(0).cpu())
    adv_img_pil.save("../results/exam_adversarial.png")

    # Visualize the noise
    noise_pil = transforms.ToPILImage()((delta * 10 + 0.5).squeeze(0).cpu())  # Amplified for visibility
    noise_pil.save("../results/perturbation_noise.png")

    print("[Success] Adversarial exam saved as 'exam_adversarial.png'")
    print("[Success] Perturbation pattern saved as 'perturbation_noise.png'")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Create a dummy exam image if you don't have one
    dummy_path = "../data/images/worksheet_000001.png"   # "dummy_exam.jpg"
    # Image.new('RGB', (1000, 1400), color='white').save(dummy_path)

    # Run the attack
    # Target: We want the model to think this exam paper is a "toaster"
    run_attack(dummy_path, target_text_str="a photo of a toaster")