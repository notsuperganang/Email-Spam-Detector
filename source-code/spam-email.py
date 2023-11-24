# %% [markdown]
# <h1>Detecting Spam Emails<h1>

# %% [markdown]
# **Importing Libraries**

# %%
# Importing necessary libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# **Loading Dataset**

# %%
data = pd.read_csv('emails.csv')
data.head()

# %% [markdown]
# 

# %% [markdown]
# **Check how many such tweets data**

# %%
data.shape

# %% [markdown]
# **Plotting the data**

# %%
sns.countplot(x='spam', data=data)
plt.show()

# %% [markdown]
# Kita dapat melihat dengan jelas bahwa jumlah sampel Ham jauh lebih banyak dibandingkan dengan Spam yang berarti bahwa dataset yang kita gunakan tidak seimbang. Sehingga perlu dilakukan downsampling agar seimbang sehingga model akan dapat mengerti lebih baik.

# %% [markdown]
# **Downsampling the dataset**

# %%
ham_msg = data[data.spam == 0]
spam_msg = data[data.spam == 1]
ham_msg = ham_msg.sample(n=len(spam_msg),
                         random_state=42)

# %% [markdown]
# **Plotting the counts of down sampled dataset**

# %%
balanced_data = pd.concat([ham_msg, spam_msg], ignore_index=True)
plt.figure(figsize=(8, 6))
sns.countplot(data=balanced_data, x='spam')
plt.title('Distribution of Ham and Spam email messages after downsampling')
plt.xlabel('Message types')


# %% [markdown]
# Meskipun menghapus data berarti kehilangan informasi, kita perlu melakukan hal ini agar data tersebut sempurna untuk dimasukkan ke dalam model pembelajaran mesin.

# %% [markdown]
# **Removing "Subject" word from text column**

# %%
balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
balanced_data.head()


# %% [markdown]
# **Removing Punctuation from Text Column**

# %%
punctuations_list = string.punctuation
def remove_punctuations(text):
	temp = str.maketrans('', '', punctuations_list)
	return text.translate(temp)

balanced_data['text']= balanced_data['text'].apply(lambda x: remove_punctuations(x))
balanced_data.head()


# %% [markdown]
# **Removing the stopwords**

# %%
def remove_stopwords(text):
	stop_words = stopwords.words('english')

	imp_words = []

	# Storing the important words
	for word in str(text).split():
		word = word.lower()

		if word not in stop_words:
			imp_words.append(word)

	output = " ".join(imp_words)

	return output


balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopwords(text))
balanced_data.head()


# %% [markdown]
# **Train-Test Split**

# %%
#train test split
train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['text'],
													balanced_data['spam'],
													test_size = 0.2,
													random_state = 42)


# %% [markdown]
# <h3>Word to vector conversion<h3>

# %% [markdown]
# kita tidak dapat memasukkan kata-kata ke dalam model pembelajaran mesin karena model tersebut hanya bekerja pada angka. Jadi, pertama-tama, kita akan mengubah kata-kata kita menjadi vektor dengan ID token menjadi kata-kata yang sesuai dan setelah mengisinya, data tekstual kita akan sampai ke tahap di mana kita dapat memasukkannya ke dalam model.

# %% [markdown]
# **Tokenization and Padding**

# %%
# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_X)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_X)
test_sequences = tokenizer.texts_to_sequences(test_X)

# Pad sequences to have the same length
max_len = 100 # maximum sequence length
train_sequences = pad_sequences(train_sequences,
								maxlen=max_len, 
								padding='post', 
								truncating='post')
test_sequences = pad_sequences(test_sequences, 
							maxlen=max_len, 
							padding='post', 
							truncating='post')


# %% [markdown]
# <h3> Model Development and Evaluation <h3>

# %% [markdown]
# kita akan menerapkan model Sequential yang akan berisi bagian-bagian berikut:
# 
# 1. Tiga Lapisan Penyematan untuk mempelajari representasi vektor unggulan dari vektor masukan.
# 2. Lapisan LSTM untuk mengidentifikasi pola yang berguna dalam urutan.
# 3. Kemudian kita akan memiliki satu lapisan yang terhubung sepenuhnya.
# 4. Lapisan terakhir adalah lapisan keluaran yang mengeluarkan probabilitas untuk kedua kelas.

# %% [markdown]
# **Build the model**

# %%
# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                    output_dim=32, 
                                    input_length=max_len))
model.add(tf.keras.layers.LSTM(16, return_sequences=True))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Print the model summary
model.summary()


# %% [markdown]
# **Compile the model**

# %%
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
			metrics = ['accuracy'],
			optimizer = 'adam')


# %% [markdown]
# **Callbacks**

# %%
es = EarlyStopping(patience=3,
				monitor = 'val_accuracy',
				restore_best_weights = True)

lr = ReduceLROnPlateau(patience = 2,
					monitor = 'val_loss',
					factor = 0.5,
					verbose = 0)
log_dir = "logs/fit"  # Tentukan direktori untuk menyimpan log TensorBoard
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# %% [markdown]
# **Train the model**

# %%
# Train the model
history = model.fit(train_sequences, train_Y,
					validation_data=(test_sequences, test_Y),
					epochs=20, 
					batch_size=32,
					callbacks = [tensorboard_callback, lr, es]
				)


# %% [markdown]
# **Showing in TensorBoard**

# %%
%load_ext tensorboard
%tensorboard --logdir logs/fit

# %% [markdown]
# **Evaluate The Model**

# %%

test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test Loss :',test_loss)
print('Test Accuracy :',test_accuracy)


# %% [markdown]
# **Training proggress graph**

# %%
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# %% [markdown]
# **Saved the model**

# %%
model.save('Spam-Detector.h5')


