import cv2

ADDRESS = "127.0.0.1"
PORT = "8880"

capture = cv2.VideoCapture(f"rtsp://{ADDRESS}:{PORT}")
print("Press Q to quit")
while(capture.isOpened()):
    try:
        ret, frame = capture.read()
        cv2.imshow(f"Stream from {PORT}", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(e)
        break
capture.release()
cv2.destroyAllWindows()