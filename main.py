import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pygame
import os
import glob

# label ranges to music genres and associated songs
GENRE_FOLDERS = {
    'classical': 'classical',
    'hip-hop': 'hip-hop',
    'electronic': 'electronic',
    'rock': 'rock'
}

LABEL_RANGE_GENRE_MAP = {
    (0, 649): {'genre': 'classical', 'songs': []},
    (650, 714): {'genre': 'rock', 'songs': []},
    (715, 799): {'genre': 'hip-hop', 'songs': []},
    (800, 1000): {'genre': 'electronic', 'songs': []},
}

# Get songs from folders
for label_range, genre_info in LABEL_RANGE_GENRE_MAP.items():
    genre_folder = GENRE_FOLDERS.get(genre_info['genre'])
    if genre_folder:
        genre_path = os.path.join("music", genre_folder)
        mp3_files = [file for file in os.listdir(genre_path) if file.endswith('.mp3')]
        genre_info['songs'] = mp3_files

class MusicApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Music Selection App")

        self.setup_ui()

        self.image_path = None
        self.model = self.load_model()
        self.current_song = None
        self.paused = False

    # GUI function
    def setup_ui(self):
        ctk.CTkLabel(self.master, text="Yehor Lashkul task3").grid(row=0, column=0, padx=10, pady=10)

        # Section 1: Upload Image
        ctk.CTkLabel(self.master, text="Choose an image from your folder:").grid(row=1, column=0, padx=10, pady=10)
        ctk.CTkButton(self.master, text="Upload Image", command=self.upload_image).grid(row=1, column=1, padx=10, pady=10)

        # Section 2: Generate Random Image
        ctk.CTkLabel(self.master, text="Generate a random noise image:").grid(row=2, column=0, padx=10, pady=10)
        ctk.CTkButton(self.master, text="Generate Image", command=self.generate_random_image).grid(row=2, column=1, padx=10, pady=10)

        # Image Display Section
        self.image_label = tk.Label(self.master)
        self.image_label.place_forget()

        # Predict and Play Music Button
        ctk.CTkButton(self.master, text="Choose appropriate song\nfor the atmosphere", command=self.predict_and_play).grid(row=4, column=0, columnspan=2, pady=10)

        ctk.CTkLabel(self.master, text="Choose a random image:").grid(row=3, column=0, padx=10, pady=10)
        ctk.CTkButton(self.master, text="Choose Random Image", command=self.choose_random_image).grid(row=3, column=1, padx=10, pady=10)

        # Play/Pause Button
        self.play_pause_button = ctk.CTkButton(self.master, text="Pause", command=self.play_or_pause_music)
        self.play_pause_button.grid(row=5, column=0, columnspan=2, pady=5)

        # Label to display currently playing song
        self.current_song_label = ctk.CTkLabel(self.master, text="")
        self.current_song_label.grid(row=6, column=0, columnspan=2, pady=5)

        # Pygame for music playback
        pygame.init()

    def load_model(self):
        # image classification model from TensorFlow Hub
        model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
        return tf.keras.Sequential([hub.KerasLayer(model_url)])

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_image(file_path)

    def generate_random_image(self):
        random_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        random_image = Image.fromarray(random_image)
        random_image_path = "random_image.png"
        random_image.save(random_image_path)
        self.display_image(random_image_path)

    def choose_random_image(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        pictures_folder = os.path.join(current_directory, "pictures")
        image_files = glob.glob(os.path.join(pictures_folder, "*.jpg"))
        if image_files:
            random_image_path = np.random.choice(image_files)
            self.display_image(random_image_path)
        else:
            print("No image files found in the folder.")

    def display_image(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((400, 350))
        self.img_photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.img_photo)
        self.image_label.image = self.img_photo

        # Show the image label at the bottom
        self.image_label.place(relx=0.5, rely=1.0, anchor=tk.S)
        self.image_path = file_path

    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array

    def predict_image_label(self, image):
        predictions = self.model.predict(image)
        predicted_label = np.argmax(predictions[0])
        return predicted_label

    def predict_and_play(self):
        try:
            image = self.preprocess_image(self.image_path)
            predicted_label = self.predict_image_label(image)
            print(f"Predicted Label: {predicted_label}")
            self.play_music_by_label_range(predicted_label)

        except Exception as e:
            print(f"Error: {e}")

    def play_music_by_label_range(self, predicted_label):
        try:
            # Func next from LABEL_RANGE_GENRE_MAP for finding first element and find appropriate diapason
            matching_range = next(
                (label_range for label_range, genre_info in LABEL_RANGE_GENRE_MAP.items() if label_range[0] <= predicted_label <= label_range[1]),
                None
            )

            if matching_range:
                genre_info = LABEL_RANGE_GENRE_MAP[matching_range]
                music_genre = genre_info['genre']
                songs = genre_info['songs']
                print(f"Predicted Genre: {music_genre}")

                if songs:
                    song_to_play = np.random.choice(songs)
                    self.play_music(music_genre, song_to_play)
                    self.current_song_label.configure(text=f"Now playing: {music_genre} : {song_to_play}")
                else:
                    print(f"No songs available for genre: {music_genre}")
            else:
                print("No matching range found for the predicted label.")

        except Exception as e:
            print(f"Error: {e}")

    def play_music(self, genre, song):
        if self.current_song:
            pygame.mixer.music.stop()

        # Set the new song path
        song_path = os.path.join("music", genre, song)
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        self.current_song = song_path

    def play_or_pause_music(self):
        if self.current_song:
            if self.paused:
                pygame.mixer.music.unpause()
                self.paused = False
                self.play_pause_button.configure(text="Pause")
            else:
                pygame.mixer.music.pause()
                self.paused = True
                self.play_pause_button.configure(text="Play")

if __name__ == '__main__':
    root = ctk.CTk()
    app = MusicApp(root)
    root.geometry("400x610")
    root.resizable(width=False, height=False)
    root.mainloop()
