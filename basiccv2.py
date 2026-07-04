cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    # The two numbers (50, 150) are the thresholds. Adjust these to detect more/fewer edges.
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow("Basic Computer Vision :Edges Detection",edges)
    if cv2.waitkey(1) & 0xFF ==ord("q"):
        break
cap.release()
cv2,destroyAllWindows()
