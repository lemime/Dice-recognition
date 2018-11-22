import numpy as np
import cv2
from mss import mss
from PIL import Image

mon = {'top': 0, 'left': 0, 'width': 1000, 'height': 400}

sct = mss()

def findBlobs(img, params):

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    return str(len(keypoints))


def findDice(dice_cascade, params):

    cap = cv2.VideoCapture(0)
    cap.set(15, -9)

    while True:

        ret, img = cap.read()

        # sct.get_pixels(mon)
        # temp = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
        # img = np.asarray(temp)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dices = dice_cascade.detectMultiScale(gray, 1.3, 20)
        # dices = dice_cascade.detectMultiScale(gray, 20, 1)
        for dice in dices:
            (x, y, w, h) = dice
            roi_gray = gray[y:y + h, x:x + w]
            count = findBlobs(roi_gray, params)
            if int(count) > 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, count, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)


        cv2.imshow("Dice Finder", img)


        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.ReleaseCapture(cap)
            break

    cv2.destroyAllWindows()


def main():

    # Load OpenCV Classifier
    dice_classifier = cv2.CascadeClassifier('cascade.xml')


    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 10
    params.maxThreshold = 200

    params.filterByArea = True
    params.minArea =10

    params.filterByCircularity = True
    params.minCircularity = 0.8

    params.filterByConvexity = True
    params.minConvexity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    findDice(dice_classifier, params)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

