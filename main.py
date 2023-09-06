import tkinter as tk
import subprocess
import threading
import sys
from tkinter import PhotoImage
from PIL import Image, ImageTk
from timer import TimerApp

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Top")
        self.geometry("680x400")

        # Load the PNG image using PIL
        image = Image.open("diversasia_and_aitop.png")  # Replace "images.png" with your image file path
        photo = ImageTk.PhotoImage(image)

        self.label = tk.Label(self, image=photo, text="\nThis is the first prototype version of the Engagement App.\n We need to collect the interaction and the video data to improve the models’ performance.\n These data will solely be used for the model’s improvement and will not be shared with anyone outside the project team.\n Once the model has been trained these data will be deleted and destroyed.", compound="top")
        self.label.image = photo  # To prevent photo from being garbage-collected
        self.label.pack(pady=20)

        self.checkbox_var = tk.BooleanVar()
        self.checkbox = tk.Checkbutton(self, text="I agree", variable=self.checkbox_var)
        self.checkbox.pack()

        self.execute_button = tk.Button(self, text="Execute Program", command=self.execute_merge, state=tk.DISABLED)
        self.execute_button.pack(pady=20)

        self.checkbox_var.trace("w", self.toggle_button_state)
        
        



    def toggle_button_state(self, *args):
        if self.checkbox_var.get():
            self.execute_button.config(state=tk.NORMAL)
        else:
            self.execute_button.config(state=tk.DISABLED)

    def execute_merge(self):
        timer_thread = threading.Thread(target=self.run_timer)
        timer_thread.start()

        try:
            subprocess.run([sys.executable, "merge.py"], check=True)
        except subprocess.CalledProcessError as e:
            print("Error executing merge.py:", e)

    def run_timer(self):
        app = TimerApp()
        app.mainloop()

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
