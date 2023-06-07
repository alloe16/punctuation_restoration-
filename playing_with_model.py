import tkinter as tk
from tkinter import messagebox
import nltk
import models
import data
import main
import sys
import tensorflow as tf
import numpy as np
import pytesseract
from PIL import Image

MAX_SUBSEQUENCE_LEN = 200
model_file = r'Model_ru_punctuator_h256_lr0.02.pcl'

def to_array(arr, dtype=np.int32):
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]

def restore(text, word_vocabulary, reverse_punctuation_vocabulary, model):
    i = 0
    while True:
        string_to_punct = ''
        subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

        y = predict(to_array(converted_subsequence), model)

        string_to_punct += subsequence[0]

        last_eos_idx = 0
        punctuations = []
        for y_t in y:
            p_i = np.argmax(tf.reshape(y_t, [-1]))
            punctuation = reverse_punctuation_vocabulary[p_i]
            punctuations.append(punctuation)

            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations)

        if subsequence[-1] == data.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        for j in range(step):
            string_to_punct += (punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
            if j < step - 1:
                string_to_punct += subsequence[1+j]

        if subsequence[-1] == data.END:
            break

        i += step
    return string_to_punct

def predict(x, model):
    return tf.nn.softmax(model(x))

def process_text():
    input_text = input_text_entry.get("1.0", "end-1c")  # Get the text from the Text widget
    if len(input_text) == 0:
        messagebox.showerror("Ошибка", "Введите текст")
        return

    text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
    pauses = [float(s.replace(data.PAUSE_PREFIX,"").replace(">","")) for s in input_text.split() if s.startswith(data.PAUSE_PREFIX)]

    text_with_punct = restore(text, word_vocabulary, reverse_punctuation_vocabulary, net)

    nltk.download('punkt')
    punkt_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
    sentences = punkt_tokenizer.tokenize(text_with_punct)
    sentences = [sent.capitalize() for sent in sentences]
    uppercase_text = ' '.join(sentences)
    result_text.set(uppercase_text)


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\nskre\AppData\Local\Tesseract-OCR\tesseract.exe'
def recognize_text_from_image():
    image_path = tk.filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if not image_path:
        return

    try:
        image = Image.open(image_path)
        recognized_text = pytesseract.image_to_string(image, lang='rus')
        input_text_entry.delete("1.0", tk.END)
        input_text_entry.insert(tk.END, recognized_text)
    except Exception as e:
        messagebox.showerror("Ошибка", "Не удалось распознать текст с изображения:\n" + str(e))

if __name__ == "__main__":
    vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, main.MINIBATCH_SIZE)).astype(int)

    print("Loading model parameters...")
    net, _ = models.load(model_file, x)

    print("Building model...")

    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    reverse_word_vocabulary = {v: k for k, v in word_vocabulary.items()}
    reverse_punctuation_vocabulary = {v: k for k, v in punctuation_vocabulary.items()}
    for key, value in reverse_punctuation_vocabulary.items():
        if value == '.PERIOD':
            reverse_punctuation_vocabulary[key] = '.'
        if value == ',COMMA':
            reverse_punctuation_vocabulary[key] = ','
        if value == '?QUESTIONMARK':
            reverse_punctuation_vocabulary[key] = '?'

    root = tk.Tk()
    root.title("Корректировка текста")

    input_label = tk.Label(root, text="Введите текст без знаков пунктуации:", font=('Arial', 12))
    input_label.pack()

    input_text_entry = tk.Text(root, width=60, height=10, font=('Arial', 12))
    input_text_entry.pack()


    def paste_text():
        input_text_entry.insert(tk.INSERT, root.clipboard_get())

    context_menu = tk.Menu(root, tearoff=0)
    context_menu.add_command(label="Вставить", command=paste_text)

    def show_context_menu(event):
        context_menu.post(event.x_root, event.y_root)

    input_text_entry.bind("<Button-3>", show_context_menu)

    recognize_button = tk.Button(root, text="Распознать текст с изображения", command=recognize_text_from_image)
    recognize_button.pack()

    process_button = tk.Button(root, text="Обработать", command=process_text)
    process_button.pack()

    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text, wraplength=400, font=('Arial', 12))
    result_label.pack()

    root.mainloop()
