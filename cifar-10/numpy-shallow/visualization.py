import matplotlib.pyplot


def show_image(img):
    fig, ax = matplotlib.pyplot.subplots()
    ax.imshow(img, interpolation=None)
    matplotlib.pyplot.show()
