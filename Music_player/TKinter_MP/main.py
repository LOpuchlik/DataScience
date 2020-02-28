from tkinter import *

root = Tk()
root.geometry('550x300')
root.title("Jagoda's Music Player")
root.iconbitmap(r'main_icon.ico')
text = Label(root, text = "Let's listen to some music!!!").pack()

# use opacity alpha values from 0.0 to 1.0
# opacity/tranparency applies to image and frame
root.wm_attributes('-alpha', 0.8)


play_pic = PhotoImage(file ='play-button.png')
play_pic = play_pic.zoom(5) #with 250, I ended up running out of memory
play_pic = play_pic.subsample(55) #mechanically, here it is adjusted to 32 instead of 320
# play_label = Label(root, image = play_pic).pack()

def play():
    print("Music plays!!")
play_button = Button(root, image = play_pic, command = play).pack(side = LEFT)


pause_pic = PhotoImage(file ='pause.png')
pause_pic = pause_pic.zoom(5) #with 250, I ended up running out of memory
pause_pic = pause_pic.subsample(55) #mechanically, here it is adjusted to 32 instead of 320

def pause():
    print('Music paused!')
pause_button = Button(root, image = pause_pic, command = pause).pack(side = LEFT)

stop_pic = PhotoImage(file ='stop.png')
stop_pic = stop_pic.zoom(5) #with 250, I ended up running out of memory
stop_pic = stop_pic.subsample(55) #mechanically, here it is adjusted to 32 instead of 320

def stop():
    print('Music stopped!')
stop_button = Button(root, image = stop_pic, command = stop).pack(side = LEFT)

root.mainloop()
