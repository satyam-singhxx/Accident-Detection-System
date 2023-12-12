import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video = cv2.VideoCapture("cars.mp4")  # Use 0 for the default camera
    if not video.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Ensure the frame is not empty
        if frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(gray_frame, (250, 250))

            pred, prob = model.predict_accident(roi[np.newaxis, :, :])
            if pred == "Accident":
                prob = round(prob[0][0] * 100, 2)

                # to beep when alert:
                # if prob > 90:
                #     os.system("say beep")

                cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
                cv2.putText(frame, f"{pred} {prob}", (20, 30), font, 1, (255, 255, 0), 2)

            cv2.imshow('Video', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()
