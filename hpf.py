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


#nambah kernel
#HPF 1 delapan tetangga
hpf1 = np.array([
    [-1, -1, -1],
    [-1, 8, -1], 
    [-1, -1, -1]
], dtype=np.float32)

#HPF 2 empat tetangga
hpf2 = np.array([
    [0, -1, 0],
    [-1, 5, -1], 
    [0, -1, 0]
    ], dtype=np.float32)

#HPF 3 bobot diagonal
hpf3 = np.array([
    [1, -2, 1],
    [-2, 5, -2], 
    [1, -2, 1]
    ], dtype=np.float32)

#meneraplpan high pass filter
J1_float = cv2.filter2D(
   noisy, cv2.CV_32F, hpf1, 
   borderType=cv2.BORDER_CONSTANT
   )

J2_float = cv2.filter2D(
   noisy, cv2.CV_32F, hpf2, 
   borderType=cv2.BORDER_CONSTANT
   )

J3_float = cv2.filter2D(
   noisy, cv2.CV_32F, hpf3, 
   borderType=cv2.BORDER_CONSTANT
   )

J1 = np.clip(np.rint(J1_float), 0, 255).astype(np.uint8)
J2 = np.clip(np.rint(J2_float), 0, 255).astype(np.uint8)
J3 = np.clip(np.rint(J3_float), 0, 255).astype(np.uint8)

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
plt.imshow(J1, cmap="gray", vmin=0, vmax=255)
plt.title("Citra HPF 1")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(J2, cmap="gray", vmin=0, vmax=255)
plt.title("Citra HPF 2")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(J3, cmap="gray", vmin=0, vmax=255)
plt.title("Citra HPF 3")
plt.axis("off")

plt.tight_layout()
plt.show()