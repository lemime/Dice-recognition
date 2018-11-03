import cv2

def findDice(dice_cascade):

    cap = cv2.VideoCapture(0)
    cap.set(15, -9)

    while True:

        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dices = dice_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in dices:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Dice Finder", img)


        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.ReleaseCapture(cap)
            break

    cv2.destroyAllWindows()


dice_cascade = cv2.CascadeClassifier('cascade.xml')
findDice(dice_cascade)

cv2.waitKey(0)
cv2.destroyAllWindows()