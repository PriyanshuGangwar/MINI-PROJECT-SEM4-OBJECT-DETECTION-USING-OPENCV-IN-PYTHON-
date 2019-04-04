import cv2 

face_data = cv2.CascadeClassifier("face_data.xml")
eyes_data = cv2.CascadeClassifier("eyes_data.xml")

cap = cv2.VideoCapture(0) 


while(1):
	ret,img=cap.read()

	gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_data.detectMultiScale(gray_img,1.05,5)

	for x,y,w,h in faces:
	    img= cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
	    
	    eyes=eyes_data.detectMultiScale(gray_img)
	    for ex,ey,ew,eh in eyes:
	    	img= cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,170,0),2)

        cv2.imshow("image",img)
        k=cv2.waitKey(1)
        if k == 27: 
            break


cv2.destroyAllWindows()
