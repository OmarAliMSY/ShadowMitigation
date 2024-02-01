from ultralytics import YOLO


class CloudClassifier:

    def __init__(self,image,model=YOLO(r'C:\Users\Omar\Documents\ShadowMitigation\shamit\cloud_classifier\best.pt')) -> None:
        self.model = model
        self.image = image
        self.class_dict = {0:"cirriform clouds",1:"high cumuliform clouds",2:"stratocumulus clouds",
                           3:"cumulus clouds",4:"cumulonimbus clouds",5:"stratiform clouds",6:"clear sky"}
    

    def inference(self):
        results = self.model(self.image,show=False)
        
        for r in results:
            prob = r.probs.cpu()
            return self.class_dict[prob.top1]

