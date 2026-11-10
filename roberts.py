import cv2
import numpy as np
import matplotlib.pyplot as plt

# membaca gambar
img = cv2.imread("foto/living.jpg")

if img is None:
    raise FileNotFoundError(
        "Citra tidak ditemukan. "
        "Pastikan folder 'foto' berada di direktori kerja"
    )

# konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# deteksi tepi roberts
kernel_roberts_x = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

kernel_roberts_y = np.array([
    [0, 1],
    [-1, 0]
], dtype=np.float32)

roberts_x = cv2.filter2D(
    gray, cv2.CV_32F, kernel_roberts_x
    )

roberts_y = cv2.filter2D(
    gray, cv2.CV_32F, kernel_roberts_y
    )

roberts = cv2.magnitude(roberts_x, roberts_y)

roberts = cv2.normalize(
    roberts, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# menampilkan hasil (2 baris x 3 kolom)
plt.figure(figsize=(14, 8))

plt.subplot(1, 5, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(gray, cmap="gray")
plt.title("Citra Grayscale")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(roberts_x, cmap="gray")
plt.title("Citra Roberts X")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(roberts_y, cmap="gray")
plt.title("Citra Roberts Y")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(roberts, cmap="gray")
plt.title("Citra Roberts (Magnitudo)")
plt.axis("off")

plt.tight_layout()
plt.show()