import cv2

classificador = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
# primeiro criar um classificador associando um arquivo xml haarcascade que já está treinado para detectar faces.

imagem = cv2.imread('pessoas/pessoas3.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
# imagem em escala de cinza serve para fazer a detecção.

facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.08, minNeighbors=11, minSize=(30, 30))
# scaleFactor serve para configuração da escala da imagem. Precisa definir ele com um valor default (o padrão é 1.1).
# minNeighbors é usado para o distânciamento entre os paramêtros.
# minSize é usado para o tamanho minímo do paramêtro (o padrão é 30x30).
# É usado quando tem imagens próximas da camêra e imagens longe.
print(len(facesDetectadas))
# O valor foi 4 porque é a quantidade de faces detectadas.

print(facesDetectadas)
# A matriz que aparece possui as listas com as informações de cada face detecada, por isso aparecem 4 listas dentro
# da matriz.

for (x, y, l, a) in facesDetectadas:
    print(x, y, l, a)
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
    # O (x, y) é a posição original (onde inicia a face que ele detectou).
    # O (x + l, y + a) é para indicar o quanto quer desenhar da borda, se colocar somente (x, y) irá aparecer somente
    # dois pontos, se somar x + l ira desenhar uma reta na horizontal (porque o l é o valor para a largura), da mesma
    # maneiro o y + a irá desenhar uma reta na vertical.
    # O (0, 0, 255) é o valor bgr da borda que irá aparecer em volta da face.
    # O 2 é a largura da borda que irá aparecer em volta da face.

cv2.imshow("Faces encontradas", imagem)
cv2.waitKey()
