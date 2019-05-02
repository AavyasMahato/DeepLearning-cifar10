# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:22:00 2018

@author: Hp
"""
from tkinter import *

root = Tk()
root.geometry('300x200')

listbox = Listbox(root)
listbox.pack()

listbox.insert(END, "Prediction Probablities :")
class_info = ["one", "two", "three", "four"]
prediction_prob = ['1','2','3','4']

for cs_name in class_info:
    listbox.insert(END, cs_name)
 

mainloop()