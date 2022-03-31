from tkinter import (LEFT, TOP, Button, Canvas, E, Frame, Label, Tk, W,
                     filedialog)

import cv2
from domain.recognition import recognition
from PIL import Image, ImageTk


class App(Tk):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.title("Picture Style Classification.")
        self.attributes("-zoomed", True)

        self.interface_frame = Frame(self)
        self.canvas_frame = Frame(self)

        self.interface_frame.pack(side=TOP, anchor=W)
        self.canvas_frame.pack(side=TOP, anchor=W)

    # ========= UX =========

        self.label = Label(
            self.interface_frame,
            text="Let's classify a picture",
            font="Times 20"
        )
        self.label.grid(row=0, column=0, padx=15, pady=7)

        self.photo = Button(
            self.interface_frame,
            text="Upload",
            width=20,
            command=lambda: self.upload_image()
        )
        self.photo.grid(row=1, column=0, padx=15, pady=7)

        self.recognition = Button(
            self.interface_frame,
            text="Classify",
            width=20,
            command=lambda: self.classifier()
            )
        self.recognition.grid(row=1, column=1, padx=15, pady=7)


    # ========= Canvas =========
        self.canvas = Canvas(self.canvas_frame, width=2000, height=900)
        self.canvas.pack(side=TOP)

        self.photo_label = Label(self.canvas, text="Photo: ", font="20")
        self.photo_label.pack(side=LEFT, anchor=E)

        self.canvas.create_window(100, 100, window=self.photo_label)

        self.hog_label = Label(self.canvas, text="Hog: ", font="20")
        self.hog_label.pack(side=LEFT, anchor=E)

        self.canvas.create_window(440, 100, window=self.hog_label)


        self.orb_label = Label(self.canvas, text="ORB: ", font="20")
        self.orb_label.pack(side=LEFT, anchor=E)

        self.canvas.create_window(790, 100, window=self.orb_label)


        self.sift_label = Label(self.canvas, text="SIFT: ", font="20")
        self.sift_label.pack(side=LEFT, anchor=E)   

        self.canvas.create_window(1150, 100, window=self.sift_label)

        self.result_label = Label(self.canvas, text="Style:", font="20")
        self.canvas.create_window(1400, 100, window=self.result_label)

        self.num_label = Label(self.canvas, text="", font="20")
        self.canvas.create_window(1500, 100, window=self.num_label)


        self.image = None
        self.result = []


    def upload_image(self) -> None:
        """
        Загрузить изображение.
        """
        global image
        filename = filedialog.askopenfilename(title="upload")
        self.image = cv2.imread(filename)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(self.image)
        image.save("./data/uploaded/photo.png")
        image = ImageTk.PhotoImage(image)
        self.canvas.create_image(100, 300,image=image)

    def classifier(self) -> None:
        """
        Запустить классификатор и отобразить результат.
        """
        if self.image is None:
            raise Exception("Image not uploaded")
        mark, desc_images = recognition([self.image])

        res_posx = 450
        res_posy = 300

        self.num_label.config(text=mark)

        for im in desc_images:
            desc = Image.open(im)
            desc = desc.resize((270, 270))
            desc = ImageTk.PhotoImage(desc)
            self.result.append(desc)
            self.canvas.create_image(res_posx, res_posy, image=desc)

            res_posx += 350
