import tkinter as tk
import tkinter.messagebox
from glob import glob
from PIL import ImageTk, Image
import shutil
import os

class ImageViewer():

    def __init__(self, window, Images):
        self.canvas = tk.Canvas(window, width=600, height=600)
        self.canvas.grid(row=0, column=0)

        self.previous_button = tk.Button(window, text = "previous image", command=self.to_previous)
        self.previous_button.place(x=50, y = 450)

        self.next_button = tk.Button(window, text = "next image", command=self.to_next)
        self.next_button.place(x=150, y = 450)

        self.Images = Images
        self.threshold = len(Images) -1
        self.img = None
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.image_idx = 0
        self.img = ImageTk.PhotoImage(file=self.Images[self.image_idx])
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)

        self.radVar = tk.IntVar()
        self.radiobutton_cat1 = tk.Radiobutton(window, text="cats", variable=self.radVar, value =0)
        self.radiobutton_cat1.place(x = 50, y = 500)
        self.radiobutton_cat2 = tk.Radiobutton(window, text="dogs", variable=self.radVar, value =1)
        self.radiobutton_cat2.place(x = 150, y = 500)
        self.labeling= tk.Button(window, text = "Label image", command=self.label_image)
        self.labeling.place(x = 250, y = 500)

        self.in_path = '../data/labeling_scheduled/'
        #self.out_path = "../data/new_data/"
        self.out_path = "../data/dl/"

    def to_previous(self):
        if len(self.Images) == 0:
            return 
        self.image_idx = (self.image_idx -1 if self.image_idx > 0  else self.threshold)
        #print(self.Images[self.image_idx])
        self.img = ImageTk.PhotoImage(file=self.Images[self.image_idx])
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        pass

    def to_next(self):
        if len(self.Images) == 0:
            return 
        self.image_idx = (self.image_idx + 1 if self.image_idx < self.threshold  else 0)
        #print(self.Images[self.image_idx])
        self.img = ImageTk.PhotoImage(file=self.Images[self.image_idx])
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)
        pass

    def label_image(self):
        if len(self.Images) == 0:
            self.canvas.delete('all')
            tk.messagebox.showinfo(title="Labeling Done", message="No unlabelled image left")
            return    
        cur_img = self.Images[self.image_idx]
        if self.radVar.get() == 0:
            shutil.move(cur_img, self.out_path+'/cats/'+cur_img.split(os.sep)[-1])
        elif self.radVar.get() == 1:
            shutil.move(cur_img, self.out_path+'/dogs/'+cur_img.split(os.sep)[-1])
        self.renew_list()

    def renew_list(self):
        files = glob(self.in_path + '*.jpg')
        self.Images = files
        self.threshold = len(self.Images) -1
        if len(files) == 0:
            self.canvas.delete('all')
            tk.messagebox.showinfo(title="Labeling Done", message="No unlabelled image left")
            return
        self.img = ImageTk.PhotoImage(file=self.Images[self.image_idx])
        self.canvas.itemconfig(self.image_on_canvas, image=self.img)

if __name__ == '__main__':
    files = glob('../data/labeling_scheduled/'+'*.jpg')
    window = tk.Tk()
    window.title("Images which needs labeling")
    ImageViewer(window, files)
    window.mainloop()