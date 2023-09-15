import cv2

classificadorRelogio = cv2.CascadeClassifier('cascades/relogios.xml')

imagem = cv2.imread('outros/relogio2.jpg')

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detectado = classificadorRelogio.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=3)

for(x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow("Encontrado", imagem)
cv2.waitKey()
