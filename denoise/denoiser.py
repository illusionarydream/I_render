import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import torch.nn.init as init


from model import DnCNN, DAE
from dataloader import ImagePairDataset

from skimage.metrics import structural_similarity as ssim
from PIL import Image
from time import time
import math


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2.0 / 9.0 / 64.0)).clamp_(-0.025, 0.025)
        init.constant_(m.bias, 0.0)


class Denoiser:
    def __init__(self, model_name, device=None):
        self.model_name = model_name.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_name == "dncnn":
            self.model = DnCNN(channels=3, num_of_layers=7).to(self.device)
        elif self.model_name == "dae":
            self.model = DAE().to(self.device)
        else:
            raise NotImplementedError(f"Model '{self.model_name}' not implemented")

        self.model.apply(weights_init_kaiming)

    def prepare_dataloader(self, train_file_path, batch_size=4, val_ratio=0.1):
        """
        Prepare the DataLoader for training and validation datasets.
        :param train_file_path: Path to the file containing training image pairs.
        :param batch_size: Batch size for training.
        :param val_ratio: Ratio of validation set size to total dataset size.
        """
        transform = ToTensor()
        full_dataset = ImagePairDataset(train_file_path, transform=transform)
        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    def train(self, num_epochs=100, learning_rate=1e-3, eval_every=10):
        """
        Train the Denoiser model.
        :param num_epochs: Number of epochs to train.
        :param learning_rate: Learning rate for the optimizer.
        :param eval_every: Frequency of evaluation on the validation set.
        :param eval_dir: Directory to save evaluation outputs.
        """
        # criterion = nn.MSELoss()
        criterion = nn.MSELoss(reduction="sum")
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        print(f"Training {self.model_name} model on {self.device} for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for noisy, clean in self.train_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                optimizer.zero_grad()
                output = self.model(noisy)
                # loss = criterion(output, clean)
                loss = criterion(output, clean) / (2 * noisy.size(0))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"[{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

            if (epoch + 1) % eval_every == 0:
                print(f"Evaluating on validation set at epoch {epoch + 1}...")
                self.evaluate_validation()

    def calculate_psnr(self, output, target):
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio) between output and target images.
        :param output: Output image tensor.
        :param target: Target image tensor.
        :return: PSNR value.
        """
        mse = nn.functional.mse_loss(output, target)
        if mse == 0:
            return float("inf")
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr.item()

    def calculate_ssim(self, output, target):
        """
        Calculate SSIM (Structural Similarity Index) between output and target images.
        :param output: Output image tensor.
        :param target: Target image tensor.
        :return: SSIM value.
        """

        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        target_np = target.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_np = (output_np * 255).astype("uint8")
        target_np = (target_np * 255).astype("uint8")
        ssim_value = ssim(output_np, target_np, channel_axis=-1, data_range=255)
        return ssim_value

    def evaluate_validation(self):
        """
        Evaluate the model on the validation set
        And return the average PSNR and SSIM.
        """
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0

        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                output = self.model(noisy)
                psnr = self.calculate_psnr(output, clean)
                ssim = self.calculate_ssim(output, clean)

                total_psnr += psnr
                total_ssim += ssim
                count += 1

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

    def denoise_single_image(self, image_path, save_path=None):
        """
        Perform denoising on a single image.
        :param image_path: Path to the noisy image.
        :param save_path: Path to save the denoised image. If None, the image will not be saved.
        :return: Denoised image tensor.
        """
        self.model.eval()
        transform = ToTensor()

        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(self.device)  # [1, C, H, W]

        start_time = time()
        with torch.no_grad():
            output = self.model(input_tensor)
        end_time = time()
        print(f"Denoising completed in {end_time - start_time:.4f} seconds")

        if save_path:
            save_image(output, save_path)
            print(f"Denoised image saved to {save_path}")

        return output.squeeze(0).cpu()  # 返回去噪后的 tensor (C, H, W)

    def save_model(self, path="dncnn.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path="dncnn.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def export_model(self, path="dncnn.pt"):
        x = torch.ones(1, 3, 64, 64).to(self.device)
        traced_model = torch.jit.trace(self.model, x)
        traced_model.save(path)
        print(f"Model exported to {path}")


def train_main(model_type="dncnn"):
    train_file_path = "datasets/local/train_list.txt"
    denoiser = Denoiser(model_name=model_type)
    denoiser.prepare_dataloader(train_file_path, batch_size=4, val_ratio=0.1)
    denoiser.train(
        num_epochs=100,
        learning_rate=2e-4,
        eval_every=5,
    )
    denoiser.save_model(f"log/{model_type}.pth")


def test_single_image(model_type="dncnn"):
    denoiser = Denoiser(model_name=model_type)
    denoiser.load_model(f"log/{model_type}.pth")
    denoiser.denoise_single_image(
        "images/denoise_.png",
        save_path="log/results/denoised_result.png",
    )
    denoiser.export_model(f"log/{model_type}.pt")


if __name__ == "__main__":
    model_type = "dncnn"
    # train_main(model_type=model_type)
    test_single_image(model_type=model_type)
