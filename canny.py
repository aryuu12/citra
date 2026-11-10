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

#deteksi tepi canny
canny1 = cv2.Canny(
    gray, 30, 80
    )

canny2 = cv2.Canny(
    gray, 100, 200
    )

canny3 = cv2.Canny(
    gray, 150, 250
    )

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
plt.imshow(canny1, cmap="gray")
plt.title("Citra Canny 1")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(canny2, cmap="gray")
plt.title("Citra Canny 2")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(canny3, cmap="gray")
plt.title("Citra Canny 3")
plt.axis("off")

plt.tight_layout()
plt.show()