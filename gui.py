import tkinter as tk


class Gui:
    def __init__(self, master, input_text):
        self.master = master
        self.input_text = input_text
        self.frame_title = tk.Frame(master)
        self.frame_body = tk.Frame(master)

        self.label_header = tk.Label(
            master=self.frame_title,
            text="Real or fake news",
            width=100,
        )
        self.label_header.pack()

        self.entry_text = tk.Text(
            master=self.frame_body,
            height=25,
        )
        self.entry_text.pack(pady=10)

        self.button = tk.Button(
            master=self.frame_body,
            text='Predict',
            width=25,
            height=5,
            command=lambda: self.buttonpress()
        )
        self.button.pack()

        self.frame_title.pack()
        self.frame_body.pack(pady=10)

    def buttonpress(self):
        self.input_text = self.entry_text.get("1.0", 'end-1c')
        return self.input_text


news = 'hej'
root = tk.Tk()

gui = Gui(root, news)

# Loopen stannar inte efter man stängt fönster :(
LOOP_ACTIVE = True
while LOOP_ACTIVE:
    root.update()
    root.update_idletasks()
    if news == gui.input_text:
        continue
    news = gui.input_text
    print(news)
