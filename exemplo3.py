import cv2

video = cv2.VideoCapture(0)
# 0 é o número padrão da camêra do notbook, se tiver outra camera precisa por o id dela.
classificadorFace = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

while True:
    conectado, frame = video.read() #metódo que vai conectar a webcam
    #A variavel conectado serve para mostrar True ou False sobre se ele conectou ou não com webcam.
    #A varaivel frame vai mostrar matrizes relacionados aos dados que estão sendo capturados pela webcam.

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, minSize=(100, 100))
    for (x, y, l, a) in facesDetectadas:
        face = cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiao = face[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.08, minNeighbors=3, minSize=(40, 40))
        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

    cv2.imshow('Vídeo', frame)

    if cv2.waitKey(1) == ord('q'):
        break
        #O 1 dentro do () significa que o waitKey vai receber 1 valor.
        #O ord('q') faz com que o q se transforme pela tabela asc e ao clicar nele a camêra fecha

video.release()
cv2.destroyAllWindows()
