import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
from tqdm import tqdm


def compute_metrics(gt_dir, test_dir):
    prefixes = ["armadillo", "tyra", "twobunny"]
    psnr_dict = defaultdict(list)
    ssim_dict = defaultdict(list)

    all_files = os.listdir(gt_dir)
    png_files = [f for f in all_files if f.endswith(".png")]

    for fname in tqdm(png_files, desc="Processing images"):
        for prefix in prefixes:
            if fname.startswith(prefix):
                gt_path = os.path.join(gt_dir, fname)
                test_path = os.path.join(test_dir, fname)

                if not os.path.exists(test_path):
                    print(f"Missing {test_path}")
                    continue

                gt_img = cv2.imread(gt_path)
                test_img = cv2.imread(test_path)

                if gt_img is None or test_img is None:
                    print(f"Cannot read {fname}")
                    continue

                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255.0
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB) / 255.0

                psnr_val = psnr(gt_img, test_img, data_range=1.0)
                ssim_val = ssim(gt_img, test_img, channel_axis=-1, data_range=1.0)

                psnr_dict[prefix].append(psnr_val)
                ssim_dict[prefix].append(ssim_val)
                break

    print("\n=== Average PSNR / SSIM per Scene ===")
    for prefix in prefixes:
        psnr_avg = sum(psnr_dict[prefix]) / len(psnr_dict[prefix]) if psnr_dict[prefix] else 0
        ssim_avg = sum(ssim_dict[prefix]) / len(ssim_dict[prefix]) if ssim_dict[prefix] else 0
        print(f"{prefix}: PSNR = {psnr_avg:.2f}, SSIM = {ssim_avg:.4f}")


# 使用示例
if __name__ == "__main__":
    gt_folder = "/home/illusionary/文档/计算机图形学/Rendering/I_render/log/clean"
    test_folder = "/home/illusionary/文档/计算机图形学/Rendering/I_render/log/dncnn10"
    compute_metrics(gt_folder, test_folder)
