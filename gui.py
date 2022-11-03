import tkinter as tk

checked = False


def buttonpress():
    print(checked)


def checkboxpress():
    global checked
    checked = not checked


def gui(b_func, c_func):
    window = tk.Tk()

    frame_title = tk.Frame(bg="black")
    frame_body = tk.Frame(bg='blue')

    label_header = tk.Label(
        master=frame_title,
        text="Hej Gabriel",
        foreground="white",
        background="red",
        width=25,
        height=10
    )
    label_header.pack()

    button = tk.Button(
        master=frame_body,
        text='Click this!',
        width=25,
        height=5,
        bg="blue",
        fg="yellow",
        command=b_func
    )
    button.pack()

    checkbox_st = tk.Checkbutton(
        frame_body,
        text='Sore Throat',
        command=c_func)
    checkbox_st.pack()

    frame_title.pack()
    frame_body.pack()

    window.mainloop()


gui(buttonpress, checkboxpress)
