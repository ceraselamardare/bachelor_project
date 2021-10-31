from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedStyle
from tkinter.filedialog import askopenfilename
import tkinter.font as font

from PIL import ImageTk,Image
from gender_age.predict_image import predict_gender, predict_age, predict_hair, predict_angle, predict_angle_v2


class GUIPredict:
    filepath_photo = None
    prediction = ''
    upload_image_first = r"C:\Users\Cera\PycharmProjects\pythonProject\gender_age\GUI\GUI.png"
    upload_image_second = r"C:\Users\Cera\PycharmProjects\pythonProject\gender_age\GUI\browse.PNG"
    exit_photo = r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\GUI\exit.PNG'

    def process_image(self):
        filepath_photo = askopenfilename()
        if filepath_photo:
            angle = predict_angle_v2(filepath_photo)
            self.filepath_photo = r'C:\Users\Cera\PycharmProjects\pythonProject\gender_age\GUI\new_img.png'
            # self.prediction = predict_gender(self.filepath_photo) + ' ' + predict_age(self.filepath_photo) + ' ' + predict_hair(self.filepath_photo)
            self.prediction = 'The person in the picture above is a {gender}, aged {age}, has {hair} and a face orientation of {angle} degree.'.format(gender=predict_gender(filepath_photo),
                                                                          age=predict_age(filepath_photo), hair=predict_hair(filepath_photo), angle=angle)
            self.__init__(root)

    def __init__(self, root):
        style = ThemedStyle(root)
        style.theme_use('arc')

        widget_list = self.all_children(root)
        for item in widget_list:
            item.pack_forget()
        root.title('Description of portraits')


        if self.filepath_photo:
            self.my_img = Image.open(self.filepath_photo)
            self.my_img = self.my_img.resize((1024, 624), Image.ANTIALIAS)
            # self.my_img = self.my_img.resize((224, 224), Image.ANTIALIAS)
            self.my_img = ImageTk.PhotoImage(self.my_img)
            my_label = ttk.Label(image=self.my_img)
            my_label.pack()

            l = ttk.Label(root, text="Image description",font=font.Font(family='Times', size='16', weight='bold'), borderwidth=0)
            T = ttk.Label(root, text=self.prediction,font=font.Font(family='Times', size='14'), borderwidth=0)
            l.pack(padx=5, pady=5)
            T.pack()

            picture = PhotoImage(file=self.upload_image_second)
            button_poza = Button(root, image=picture,compound=BOTTOM,borderwidth=0, highlightthickness=0,
                                     text='',
                                     command=self.process_image)
            button_poza.image = picture
            button_poza.pack(padx=10, pady=10, ipady=5, ipadx=5)
            button_poza.bind('<Button-1>', lambda *_: self.process_image())
        else:
            picture = PhotoImage(file = self.upload_image_first).subsample(2)
            button0 = Button(root, image=picture, compound=BOTTOM,borderwidth=0, highlightthickness=0, command=self.process_image)
            button0.image = picture

            button0.pack()
            button0.bind('<Button-1>', lambda *_: self.process_image())


        picture = PhotoImage(file=self.exit_photo)
        button_quit = Button(root, text="", image=picture, compound=BOTTOM,borderwidth=0, highlightthickness=0, command=root.quit)
        button_quit.image = picture
        button_quit.pack()
        button_quit.bind('<Button-1>', lambda *_: root.quit())

    def all_children(self, window):
        _list = window.winfo_children()

        for item in _list:
            if item.winfo_children():
                _list.extend(item.winfo_children())

        return _list

root = Tk()

my_gui = GUIPredict(root)
root.mainloop()
