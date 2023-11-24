## Detektor Email Spam

Repositori ini berisi model untuk deteksi email spam yang diimplementasikan menggunakan Keras dan TensorFlow.

## Dataset

Dataset (`emails.csv`) tersedia dalam direktori `dataset`. Ini mencakup teks email dan label yang menunjukkan apakah email tersebut adalah spam (1) atau bukan spam (0).

## Instalasi

1. Pastikan Anda memiliki Python terinstal.
2. Instal dependensi yang diperlukan dengan menjalankan:

```bash
pip install -r requirements.txt
```

## Penggunaan

1. Unduh dataset email (`emails.csv`) dan letakkan di dalam direktori `dataset`.
2. Buka dan jalankan notebook `spam-email-detector.ipynb` atau jalankan skrip `spam-email-detector.py` untuk melatih dan mengevaluasi model.
3. Jika Anda ingin menggunakan model yang sudah dilatih, pastikan untuk menyertakan instruksi cara mengunduh atau memuatnya.

## Struktur Proyek

- `dataset/`: Direktori yang berisi dataset email (`emails.csv`).
- `requirements.txt`: Berkas yang berisi daftar dependensi.
- `source-code/`: Direktori yang berisi file kode sumber.
  - `spam-email-detector.ipynb`: Notebook Jupyter untuk melatih dan mengevaluasi model.
  - `spam-email-detector.py`: Skrip Python untuk tujuan yang sama.

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan ikuti langkah-langkah berikut:
1. Fork proyek.
2. Buat branch baru (`git checkout -b fitur-baru`).
3. Lakukan perubahan dan commit (`git commit -m 'Menambahkan fitur baru'`).
4. Push ke branch (`git push origin fitur-baru`).
5. Buat pull request.

## Lisensi

Proyek ini dilisensikan di bawah [Lisensi MIT](LICENSE).
```
