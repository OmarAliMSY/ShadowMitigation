import cv2
import pathlib



def show_image(image,
               title=None):
    """
    Show image in a new window
    :param image: image to show
    :param title: title of the window
    :return: None
    """
    if title is None:
        title = "Image"
    if type(image) is str or type(image) is pathlib.WindowsPath:
        image = cv2.imread(image)
        image = cv2.resize(image,(1280,640))

        cv2.imshow(title, image)
    cv2.waitKey(2)

