import cv2

cap = cv2.VideoCapture(0)

input_image = cv2.imread('glasses.png')
img1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    cap.set(3, 960)
    cap.set(4, 540)

    frame = cv2.flip(frame, 1)
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])

    print(len(good))
    # Draw first 10 matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,None, flags=2)
    cv2.imshow('frame', img3)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
