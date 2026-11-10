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

# deteksi tepi prewitt
kernel_prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

kernel_prewitt_y = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
], dtype=np.float32)

prewitt_x = cv2.filter2D(
    gray, cv2.CV_32F, kernel_prewitt_x
    )

prewitt_y = cv2.filter2D(
    gray, cv2.CV_32F, kernel_prewitt_y
    )

prewitt = cv2.magnitude(prewitt_x, prewitt_y)

prewitt = cv2.normalize(
    prewitt, None, 0, 255, cv2.NORM_MINMAX
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
plt.imshow(prewitt_x, cmap="gray")
plt.title("Citra Prewitt X")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(prewitt_y, cmap="gray")
plt.title("Citra Prewitt Y")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(prewitt, cmap="gray")
plt.title("Citra Prewitt (Magnitudo)")
plt.axis("off")

plt.tight_layout()
plt.show()