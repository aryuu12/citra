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

# deteksi tepi sobel
sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

sobel = cv2.magnitude(sobel_x, sobel_y)

# normalisasi untuk visualisasi
sobel = cv2.normalize(
    sobel, None, 0, 255, cv2.NORM_MINMAX
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
plt.imshow(sobel_x, cmap="gray")
plt.title("Citra Sobel X")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(sobel_y, cmap="gray")
plt.title("Citra Sobel Y")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(sobel, cmap="gray")
plt.title("Citra Sobel (Magnitudo)")
plt.axis("off")

plt.tight_layout()
plt.show()