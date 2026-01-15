#import opencv
import cv2

#import the video
original_can = cv2.VideoCapture("original_vid.mov")

while True:
    #get the frames, ret is a boolean: whether video is successfully read
    ret, vid = original_can.read()

    #greyscale for hough circle transform
    grey = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    #for clearer pictures
    grey = cv2.GaussianBlur(grey, (9, 9), 2)

    #hough circle transform
    processed_can = cv2.HoughCircles(
        image=grey,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1300,
        param1=60,
        param2=70,
        minRadius=200,
        maxRadius=240
    )

    #ensures that the video doesn't break if no circles detected
    if processed_can is None:
        continue

    #access to the circles' data
    processed_can = processed_can[0]

    #draw the circle
    for c in processed_can:
        x = int(c[0])
        y = int(c[1])
        r = int(c[2])
        cv2.circle(vid, (x, y), r, (255, 150, 0), 11)
        cv2.circle(vid, (x, y), 18, (0, 0, 255), -1)

    cv2.imshow("processed circle", vid)         
    cv2.waitKey(1)                              

original_can.release()
cv2.destroyAllWindows()

#1) Open the video file
#2) Read one frame at a time in a loop
#3) Convert each frame to grayscale and make it clearer
#4) Use hough circles to detect the circular can
#5) Draw the circle and its center on the frame
#6) Display the processed frame as a video
#7) Stop when the video ends or the user quits

