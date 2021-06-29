import cv2

#Our Image
#img_file = 'Car_Image.jpg'
# create opencv image
#img = cv2.imread(img_file)


#creaate opencv video
#video = cv2.VideoCapture('TeslaDashcam.mp4')
#video = cv2.VideoCapture('withpedestrians.mp4')
video = cv2.VideoCapture(0)
#Our pretrained car and pedestrian classisfier

car_tracker = cv2.CascadeClassifier('car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')


while True:

    successfull_read, frame = video.read()

    if successfull_read:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    #detect cars AND pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow('Lucifer car detector', frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break


    
video.release()






    
    
    
'''
#create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)



#convert to grascale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)

#displays coordinates of all cars
#print(cars)

#draw rectange around the cars

#car1 = cars[0]
#(x, y, w, h) = car1
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

#display images with cars spotted
cv2.imshow('Lucifer car detector', img)

cv2.waitKey()

'''
print('Code Completed')