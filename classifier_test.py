from utills import FileHandler
from shamit import CloudClassifier
from utills.CvUtils import show_image


fh = FileHandler(dataset_path=r"playground",legal_extensions=["jpg","jpeg"],foldername="random")
impath = fh.path_file()[2]
CC = CloudClassifier(image=impath)
inf = CC.inference()

print(type(impath))

show_image(impath)

print(inf)

