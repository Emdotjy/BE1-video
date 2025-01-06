import tkinter as tk
from tkinter import filedialog, messagebox
import os
from utils import convertir_video_en_array
from reconnaissance_sequence import *
from interface_reconnaissance_video import *


class VideoProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Configure the main window
        self.title("Video Processing App")
        self.geometry("600x400")

        # Create interface elements
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        self.title_label = tk.Label(self, text="Video Processing Interface", font=("Arial", 20))
        self.title_label.pack(pady=20)

        # Button to select video
        self.select_video_button = tk.Button(
            self, text="Select Video for Processing", command=self.process_video
        )
        self.select_video_button.pack(pady=10)

        # Button to select test video
        self.select_test_video_button = tk.Button(
            self, text="Select Test Video", command=self.test_video
        )
        self.select_test_video_button.pack(pady=10)

        # Status Label
        self.status_label = tk.Label(self, text="Status: Idle", font=("Arial", 14))
        self.status_label.pack(pady=20)

    def process_video(self):
        # Open file dialog to select video
        video_path = filedialog.askopenfilename(
            title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )
        if not video_path:
            return

        try:
            self.status_label.configure(text="Processing video...")

            # Convert video to array and process it
            video = convertir_video_en_array(video_path)
            video_standart = standardize_video_color(video)
            pub_list = segmentation_spot_pub(video_standart)

            # Add each segmented pub to the database
            for pub in pub_list:
                add_pub_to_bdd(pub)

            self.status_label.configure(text="Video processed and added to database.")
            messagebox.showinfo("Success", "Video processed and added to database.")
        except Exception as e:
            self.status_label.configure(text="Error during processing.")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def test_video(self):
        # Open file dialog to select test video
        test_video_path = filedialog.askopenfilename(
            title="Select Test Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )
        if not test_video_path:
            return

        try:
            self.status_label.configure(text="Testing video against database...")

            # Convert test video to array and process it
            video_test = convertir_video_en_array(test_video_path)
            video_test_standart = standardize_video_color(video_test)
            pub_list_test = segmentation_spot_pub(video_test_standart)

            for pub_test in pub_list_test:
                recognise_pub_in_bdd(pub_test)

            self.status_label.configure(text="Test video processed.")
            messagebox.showinfo("Success", "Test video processed.")
        except Exception as e:
            self.status_label.configure(text="Error during testing.")
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    app = VideoProcessingApp()
    app.mainloop()
