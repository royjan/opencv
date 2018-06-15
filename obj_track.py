import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 540)

input_image = cv2.imread('20180519_124108.jpg')
input_image = cv2.resize(input_image, (400, 300))
img1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

orb = cv2.xfeatures2d.SIFT_create()

cv2.namedWindow('LeeGonen', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('LeeGonen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

font = cv2.FONT_HERSHEY_SIMPLEX
THRESHOLD = 0.8

while True:
    ret, frame = cap.read()
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < THRESHOLD * n.distance:
                good.append([m])

        print(len(good))
        img3 = cv2.drawMatchesKnn(input_image, kp1, frame, kp2, good, None, flags=2)

        if len(good) >= 10:
            cv2.putText(img3, "it's a match!", (200, 500), font, 4, (30, 150, 50), 2, cv2.LINE_AA)

        cv2.imshow('LeeGonen', img3)
    except Exception as ex:
        print(str(ex))
        cv2.putText(frame, "searching an object", (200, 500), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('LeeGonen', frame)
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
