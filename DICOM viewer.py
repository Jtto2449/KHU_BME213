import tkinter as tk
from tkinter import filedialog
import pydicom
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

class DICOMViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM VIEWER: BME213")

        self.image = None
        self.original_image = None

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Open File", width=10, height=5, command=self.load_dicom).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Original", width=10, height=5, command=self.show_original).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Binary", width=10, height=5, command=self.apply_otsu).pack(side=tk.LEFT)
        tk.Button(self.button_frame, text="Average filter", width=10, height=5, command=self.avg_filter).pack(side=tk.LEFT)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

    def load_dicom(self):
        file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])
        if file_path:
            dicom = pydicom.dcmread(file_path)
            self.original_image = dicom.pixel_array.astype(np.float32)
            self.image = self.original_image.copy()
            self.update_canvas()

    def update_canvas(self):
        self.ax.clear()
        if self.image is not None:
            self.ax.imshow(self.image, cmap='gray')
        self.canvas.draw()

    def show_original(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.update_canvas()

    def apply_otsu(self):
        if self.image is not None:
            img_uint8 = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.image = binary
            self.update_canvas()

    def avg_filter(self):
        if self.image is not None:
            kernel = np.ones((5, 5)) / 25
            filtered = convolve(self.image, kernel)
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.image = filtered
            self.update_canvas()

if __name__ == "__main__":
    root = tk.Tk()
    app = DICOMViewer(root)
    root.mainloop()