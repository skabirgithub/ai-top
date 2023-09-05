import tkinter as tk
import threading
import time

class TimerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Timer")
        self.geometry("400x120")

        self.timer_label = tk.Label(self, text="3:00")
        self.timer_label.pack(pady=10)

        self.remaining_time = 180  # 3 minutes

        self.additional_label = tk.Label(self, text="The system is collecting data to calibrate to your environment. \nThis will take 3 minutes. \nPlease keep using the system normally and wait for the timer to expire.")
        self.additional_label.pack(pady=5)

        timer_thread = threading.Thread(target=self.start_timer)
        timer_thread.start()

    def start_timer(self):
        while self.remaining_time > 0:
            minutes = self.remaining_time // 60
            seconds = self.remaining_time % 60
            self.timer_label.config(text=f"{minutes:02}:{seconds:02}")
            time.sleep(1)
            self.remaining_time -= 1

        self.destroy()  # Close the window when the timer is done

if __name__ == "__main__":
    app = TimerApp()
    app.mainloop()
