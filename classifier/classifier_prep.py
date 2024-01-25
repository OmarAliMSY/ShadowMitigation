import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
import os
train_set = pd.read_csv(r"C:\Users\o.abdulmalik\Documents\Shadow-Mitigation\cloud-type-classification2\train.csv")


labels = train_set["label"]

labs = set(labels)
for lab in labs:
    print(lab)
    p = os.path.join(r"C:\Users\o.abdulmalik\Documents\Shadow-Mitigation\cloud-type-classification2\images\train",str(lab),"*.jpg")
    test_images = glob(p)
    filenames_without_ext = [os.path.basename(path) for path in test_images]
    x_train,x_test = train_test_split(filenames_without_ext,test_size=0.2)

    
    for file in x_test:
        test_pdir = os.path.join(r"C:\Users\o.abdulmalik\Documents\Shadow-Mitigation\cloud-type-classification2\images\test",str(lab))
        train_pdir = os.path.join(r"C:\Users\o.abdulmalik\Documents\Shadow-Mitigation\cloud-type-classification2\images\train",str(lab))
        print(test_pdir,file,lab)
        os.replace(os.path.join(train_pdir,file),os.path.join(test_pdir,file))
        
       
