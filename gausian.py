import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

#membaca citra
img = cv2.imread("foto/living.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
   raise FileNotFoundError(
      "Citra tidak ditemukan. "
      "Pastikan foto berada di folder yang sama"
      )

#menambahkan noise pada citra
noisy = img.copy()
rng = np.random.default_rng(42)
noise_mask = rng.random(img.shape[:2])
noisy[noise_mask < 0.02] = (0)
noisy[noise_mask > 0.98] = (255)

print("Ukuran citra: ", noisy.shape)
print("Tipe data citra: ", noisy.dtype)

#Menerapkan gaussian Filter
gaussian1 = cv2.GaussianBlur(
    noisy, (3, 3),
     sigmaX=1.0)
gaussian2 = cv2.GaussianBlur(
    noisy, (5, 5),
     sigmaX=1.0)
gaussian3 = cv2.GaussianBlur(
    noisy, (7, 7),
     sigmaX=1.0)


#menampilkan citra asli
plt.figure(figsize=(12, 4))

plt.subplot(1, 5, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(noisy, cmap="gray", vmin=0, vmax=255)
plt.title("Citra noisy")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(gaussian1, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Gaussian 1")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(gaussian2, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Gaussian 2")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(gaussian3, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Gaussian 3")
plt.axis("off")

plt.tight_layout()
plt.show()