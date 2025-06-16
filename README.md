
# Deteksi Berita Palsu Menggunakan Deep Learning

## 1. Latar Belakang

Di era digital saat ini, penyebaran informasi sangat cepat dan tidak terkontrol, termasuk berita yang tidak benar atau hoaks. Berita palsu dapat menyesatkan pembaca dan menyebabkan dampak negatif dalam berbagai aspek seperti politik, ekonomi, hingga keamanan publik. Oleh karena itu, dibutuhkan sistem otomatis yang mampu mendeteksi berita palsu secara efektif. Proyek ini bertujuan untuk membangun model deep learning yang mampu mengklasifikasikan berita sebagai **real** atau **fake** berdasarkan konten teksnya.

## 2. Klarifikasi Masalah

### Problem Statement
Bagaimana membangun model deep learning yang dapat mengklasifikasikan suatu berita sebagai palsu atau asli dengan akurasi tinggi berdasarkan teks berita?

### Goals
- Melakukan eksplorasi dan praproses data berita dari dua sumber: berita palsu dan berita asli.
- Membangun model deep learning (LSTM) untuk klasifikasi binary: `Fake` atau `Real`.
- Mengevaluasi model berdasarkan metrik klasifikasi seperti akurasi, precision, recall, dan F1-score.

---

## 3. Informasi Dataset

- Dataset berasal dari Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Terdiri dari dua file:
  - `Fake.csv` (berita palsu)
  - `True.csv` (berita asli)

### Jumlah Data:
- Fake: 23.481 data
- Real: 21.417 data
- Total: 44.898 data

### Struktur Kolom:
- `title`: Judul berita
- `text`: Isi utama berita
- `subject`: Topik berita
- `date`: Tanggal publikasi
- `label`: Label target (0 = fake, 1 = real)

---

## 4. Data Preparation

### Langkah yang Dilakukan:
1. **Labeling dan Penggabungan Data**  
   Menambahkan kolom `label` ke masing-masing file dan menggabungkan keduanya.

2. **Pembersihan Teks**  
   - Konversi ke huruf kecil
   - Hapus URL, HTML tag, tanda baca, angka, newline
   - Hapus kata yang mengandung angka dan simbol

3. **Tokenisasi dan Padding**  
   - Tokenisasi menggunakan Keras `Tokenizer`
   - Padding dengan panjang tetap `max_len`

4. **Pembagian Dataset**  
   - Training: 80%
   - Testing: 20%

---

## 5. Pemodelan

### Arsitektur Model:

```python
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

### Konfigurasi Pelatihan:
- Loss function: `binary_crossentropy`
- Optimizer: `adam`
- Epoch: 100
- Batch size: 64
- Validation split: 0.2

### Callback:
```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)
```

---

## 6. Evaluasi Model

### Metrik:
- **Accuracy**: 0.98
- **Precision**: 0.97
- **Recall**: 0.99
- **F1 Score**: 0.98

### Classification Report:

| Label | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Fake  | 0.89      | 0.97   | 0.93     |
| Real  | 0.98      | 0.99   | 0.98     |

### Confusion Matrix:

|                        | Predicted Fake | Predicted Real |
|------------------------|----------------|----------------|
| **Actual Fake**        | 3368 (TP)      | 90 (FN)        |
| **Actual Real**        | 34 (FP)        | 4227 (TN)      |

---

## 7. Kesimpulan

Model LSTM yang dibangun mampu mengklasifikasikan berita palsu dan asli dengan akurasi sangat tinggi. Evaluasi menunjukkan nilai F1-score sebesar 0.98, dengan kesalahan klasifikasi yang sangat rendah. Model ini menunjukkan performa yang baik dan dapat digunakan sebagai dasar untuk sistem deteksi berita hoaks secara otomatis di masa depan.