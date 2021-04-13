import tkinter as tk
import tkinter.messagebox
from glob import glob
from PIL import ImageTk, Image
import shutil
import os

class ImageViewer():

    def __init__(self, window, Images, category):

        self.window = window
        self.canvas = tk.Canvas(window, width=600, height=600)
        self.canvas.grid(row=0, column=0)

        if len(Images) == 0:
            tk.messagebox.showinfo(title="error", message="No unlabelled image")
            self.window.destroy()
            return  

        self.previous_button = tk.Button(window, text = "<< prev", command=self.to_previous)
        self.previous_button.place(x=50, y = 450)

        self.next_button = tk.Button(window, text = "next >>", command=self.to_next)
        self.next_button.place(x=150, y = 450)

        self.Images = Images
        self.threshold = len(Images) -1
        self.img = None
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.image_idx = 0
        self.img = ImageTk.PhotoImage(file=self.Images[self.image_idx])
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)

        self.category = category
        self.radVar = tk.IntVar()
        self.radioButton = []
        for i, cat in enumerate(self.category):
            self.radioButton.append(tk.Radiobutton(window, text=cat, variable=self.radVar, value =i))
            self.radioButton[i].place(x= 100*i + 50, y=500)
            
        self.labeling= tk.Button(window, text = "Label image", command=self.label_image)
        self.labeling.place(x = 250, y = 550)

        self.in_path = '../data/labeling_scheduled/'
        self.out_path = "../data/dl/"
        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path, exist_ok=True)
        

    def to_previous(self):
        if len(self.Images) == 0:
            return 
        self.image_idx = (self.image_idx -1 if self.image_idx > 0  else self.threshold)
        self.img = ImageTk.PhotoImage(file=self.Images[self.image_idx])
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        pass

    def to_next(self):
        if len(self.Images) == 0:
            return 
        self.image_idx = (self.image_idx + 1 if self.image_idx < self.threshold  else 0)
        self.img = ImageTk.PhotoImage(file=self.Images[self.image_idx])
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        pass

    def label_image(self):
        if len(self.Images) == 0:
            self.canvas.delete('all')
            tk.messagebox.showinfo(title="Labeling Done", message="No unlabelled image left")
            return    
        cur_img = self.Images[self.image_idx]

        class_name = self.category[self.radVar.get()]
        dst_dir = os.path.join(self.out_path, class_name)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        shutil.move(cur_img, os.path.join(dst_dir, cur_img.split(os.sep)[-1]))
        self.renew_list()

    def renew_list(self):
        files = glob(self.in_path + '*.jpg')
        self.Images = files
        self.threshold = len(self.Images) -1
        if len(files) == 0:
            self.canvas.delete('all')
            tk.messagebox.showinfo(title="Labeling Done", message="No unlabelled image left")
            return
        
        if self.image_idx >= self.threshold:
            self.image_idx = self.threshold
        self.img = ImageTk.PhotoImage(file=self.Images[self.image_idx])
        
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)

def labeling(scheduling_path, category):
    files = glob(os.path.join(scheduling_path, '*.jpg'))
    window = tk.Tk()
    window.title("Images which need labeling")
    window.resizable(width=False, height=False)

    ImageViewer(window, files, category)
    window.mainloop()
