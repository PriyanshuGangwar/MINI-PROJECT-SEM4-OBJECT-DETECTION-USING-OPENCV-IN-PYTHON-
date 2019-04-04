import numpy as np
from collections import OrderedDict
import cv2
import math


class centroid_tracker():

    def __init__(self):

        self.objectID = 0
        self.objects = OrderedDict()
       
    def register(self,centroid):

        self.objects[self.objectID] = centroid
        self.objectID += 1

    def ret_id(self):
        return self.objectID

    def update(self,rects):
         
        centres = np.zeros((len(rects), 2), dtype="int")
        
    	for (i, (startX, startY, endX, endY)) in enumerate(rects):

    	    cX = int((startX + endX) / 2.0)
	    cY = int((startY + endY) / 2.0)
	    centres[i] = (cX, cY)
    
        if len(self.objects) == 0:
            
            for i in range(len(centres)):
                self.register(centres[i])

        else:

            obj_id = list(self.objects.keys())
	    obj_centres = list(self.objects.values())
            
            usedrows = []
            for (ID,(x1,y1)) in zip(obj_id,obj_centres):
                D=[]
                row=[]
                
                for(x2,y2) in centres:
                    D.append(math.hypot(x2-x1, y2-y1)) 
                    row.append((x2,y2))
                  

                p=np.argmin(D) 
                usedrows.append(row[p])
                self.objects[ID] = row[p]   

            

            for (x1,y1) in centres:
                
                if (x1,y1) in usedrows:
                    continue
                else:
                    
                    self.register((x1,y1))


        
        return self.objects




face_data = cv2.CascadeClassifier("face_data.xml")

cap = cv2.VideoCapture("vid1.webm") 
ct = centroid_tracker()

while(True):
	ret,img=cap.read()

	gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_data.detectMultiScale(gray_img,1.05,5)
         
	for x,y,w,h in faces:
	    img= cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
	    
	    
        
    
        objects = ct.update(faces)
	#for (objectID, centroid) in objects.items():
            
        cv2.putText(img ,str("OBJECTS: "+str(ct.ret_id())),(10,350),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255)) 


        cv2.imshow("image",img)
        k=cv2.waitKey(1)
        if k == 27: 
            break
        

cv2.destroyAllWindows()

print "Total object detected : ",ct.ret_id()
