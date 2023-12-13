import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) #("videos/test.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

frameWidth = 1920
frameHeight = 1080
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB) #construit le maillage
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks: #pour chaque
            #Dessine le maillage sur le visage
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

    cTime = time.time() #(mili)seconde actuelle (epoch linux)
    fps = 1 / (cTime - pTime) #fps = 1 / (seconde actuelle)
    pTime = cTime #met a jour la seconde precedente a chaque
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) #affiche les FPS
    cv2.imshow("Image", img) #montre la video
    if cv2.waitKey(1) == 27:
        break #echap pour quitter
cap.release()

