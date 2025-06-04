import os

noisy_dir = "datasets/local"
clean_dir = "datasets/local"
output_list_path = "datasets/local/train_list.txt"

num_samples = 168

with open(output_list_path, "w") as f:
    for i in range(num_samples):
        noisy_path = os.path.join(noisy_dir, f"noise_{i:04d}.png")
        clean_path = os.path.join(clean_dir, f"clean_{i:04d}.png")
        f.write(f"{noisy_path},{clean_path}\n")

print(f"Saved {num_samples} pairs to {output_list_path}")
