import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

def create_real_time_dashboard(y_data1, y_data2, y_data3, y_data4, action):
    # Create a tkinter window
    root = tk.Tk()
    root.title("Engagement Dashboard")
    #root.geometry("680x400")

    # Create a frame to hold the graphs
    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    # Create a figure for each graph with the desired size (100x200)
    fig1, ax1 = plt.subplots(figsize=(4, 2))  
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 1)
    line1, = ax1.plot([], [], lw=2)
    ax1.set_title("Eye Movement")

    fig2, ax2 = plt.subplots(figsize=(4, 2))  
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1)
    line2, = ax2.plot([], [], lw=2)
    ax2.set_title("Mouse & Keyboard Interaction")

    fig3, ax3 = plt.subplots(figsize=(4, 2))  
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 1)
    line3, = ax3.plot([], [], lw=2)
    ax3.set_title("Heart Rate Deviation")


    fig4, ax4 = plt.subplots(figsize=(4, 2))  
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 1)
    line4, = ax4.plot([], [], lw=2)
    ax4.set_title("Heart Rate Deviation")


    def draw_fahad():
        # Update canvas
        canvas1.draw()
        canvas2.draw()
        canvas3.draw()

    # Function to update the data for each graph
    def update():
        x = np.linspace(0, 10, 100)
        y1 = y_data1
        y2 = y_data2
        y3 = y_data3
        y4 = y_data4

        line1.set_data(x, y1)
        line2.set_data(x, y2)
        line3.set_data(x, y3)
        line4.set_data(x, y4)

        # Schedule the update after a delay (e.g., 1000 milliseconds)
        root.after(1000, update)

    # Create canvas widgets to display the graphs in the frame
    canvas1 = FigureCanvasTkAgg(fig1, master=frame)
    canvas2 = FigureCanvasTkAgg(fig2, master=frame)
    canvas3 = FigureCanvasTkAgg(fig3, master=frame)

    canvas1.get_tk_widget().grid(row=0, column=0)
    canvas2.get_tk_widget().grid(row=1, column=0)
    canvas3.get_tk_widget().grid(row=2, column=0)

    if action == 'draw':
        # Start the update loop
        draw_fahad()
    else:
        update()

    # Start the tkinter main loop
    root.mainloop()

# Example usage:
if __name__ == "__main__":
    # Initialize data for the graphs
    y_data1, y_data2, y_data3, y_data4 = [], [], [], []

    # Pass the data to the function
    create_real_time_dashboard(y_data1, y_data2, y_data3, y_data4, action='update')
