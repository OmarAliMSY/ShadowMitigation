from utills import FileHandler
from shamit import CloudClassifier

fh = FileHandler(dataset_path=r"playground",legal_extensions=["jpg","jpeg"],foldername="random")

CC = CloudClassifier(image=fh.path_file()[3 ])
inf = CC.inference()
print(inf)


#