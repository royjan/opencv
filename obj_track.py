import cv2

cap = cv2.VideoCapture(0)

input_image = cv2.imread('image.jpg')
img1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    cap.set(3, 960)
    cap.set(4, 540)

    frame = cv2.flip(frame, 1)
    img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(input_image, kp1, frame, kp2, matches[:10], None, flags=2)
    cv2.imshow('frame', img3)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
