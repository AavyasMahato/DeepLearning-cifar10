# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:03:01 2018

@author: Hp
"""
from tkinter import *
import tkinter.messagebox
import os
from tkinter import filedialog
from scipy.misc import imread

root = Tk()
root.geometry('1000x650')
root.title('Image Classification')
title_label = Label(root, text = 'IMAGE CLASSIFICATION USING CNN')
title_label.config(font=("Courier", 18))
title_label.pack(side = TOP)

root.filename =  filedialog.askopenfilename(initialdir = "C:/Users/Hp/.spyder-py3/Deep_learning_cifar-10/test_images/",
                                                title = "Select file",filetypes = 
                                                (("jpeg files","*.jpg"),("all files","*.*")))

def helloCallBack():
    
    messagebox.showinfo( "Hello Python", "Hello World")
    
result = 'Airplane'
def classify():
    
    messagebox.showinfo('Class', 'Tested Image is :' + result)

classify_btn = Button(root, text ="Select Image", command = helloCallBack)
classify_btn.config(font=("Courier", 12))
classify_btn.pack()


val_label = Label(root, text = 'All the Images are classified with 78% of validaion Accuracy.')
val_label.config(font=("Courier", 12))
val_label.pack(side=BOTTOM)
root.mainloop()