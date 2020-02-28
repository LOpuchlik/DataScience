from tkinter import *
from pygame import mixer

root = Tk()

mixer.init()

root.geometry('550x300')
root.title("Jagoda's Music Player")
root.iconbitmap(r'main_icon.ico')
text = Label(root, text = "Let's listen to some music!!!").pack()

# use opacity alpha values from 0.0 to 1.0
# opacity/tranparency applies to image and frame
root.wm_attributes('-alpha', 0.8)

def play():
    mixer.music.load('sample.wav')
    mixer.music.play()

def stop():
    mixer.music.stop()

def adjust_volume(val):
    volume = int(val)/100
    mixer.music.set_volume(volume)



play_pic = PhotoImage(file ='play-button.png')
play_pic = play_pic.zoom(5) #with 250, I ended up running out of memory
play_pic = play_pic.subsample(55) #mechanically, here it is adjusted to 32 instead of 320
play_button = Button(root, image = play_pic, command = play).pack(side = LEFT)

stop_pic = PhotoImage(file ='stop.png')
stop_pic = stop_pic.zoom(5) #with 250, I ended up running out of memory
stop_pic = stop_pic.subsample(55) #mechanically, here it is adjusted to 32 instead of 320
stop_button = Button(root, image = stop_pic, command = stop).pack(side = LEFT)

volume_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, command = adjust_volume).pack(side = LEFT)






"""
def pause():
    mixer.music.pause()

pause_pic = PhotoImage(file ='pause.png')
pause_pic = pause_pic.zoom(5) #with 250, I ended up running out of memory
pause_pic = pause_pic.subsample(55) #mechanically, here it is adjusted to 32 instead of 320
pause_button = Button(root, image = pause_pic, command = pause).pack(side = LEFT)

"""





root.mainloop()
