import cv2



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
    cv2.imshow(title, image)
    cv2.waitKey(1)
