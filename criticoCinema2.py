from tqdm import tqdm

import matplotlib.pyplot as plt

from deepface import DeepFace

import chromadb

import cv2

import uuid

import numpy as np

from fer.fer import FER



def softmax(distances):
    return distances / np.sum(distances)



client = chromadb.PersistentClient(path="./encodingCelebridades")

coll = client.get_or_create_collection(name="celebridades")


fonte, escala = cv2.FONT_HERSHEY_SIMPLEX, 0.5
cascade_path = 'haarcascade_frontalface_default.xml'
cv2.data.haarcascades = cascade_path
detector = FER(mtcnn=False,cascade_file=cascade_path)



movingAverage = 10

n = 0

nomeTela = "Critico de Cinema"


plt.ion()

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(5, 10), num='Estatisticas')

fig.delaxes(ax2)
ax2 = fig.add_subplot(2, 1, 2, projection='polar')
barras = ax1.bar(range(5), [0.0] *5)
plt.show(block=False)
cv2.namedWindow(nomeTela, cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)


proximo,pct = "...",0
i = 0
emotionEveryNFrames = 30
bb = []

while True:

    i = (i+1)%emotionEveryNFrames
    ret, frame = cap.read()

    if not ret:
        print("Error: Falha ao ler a imagem.")
        break

    if i==0:
        detection = DeepFace.represent(img_path=frame, model_name="Facenet512",detector_backend="mtcnn", enforce_detection=False,align=True)
        imgEnc = detection[0]['embedding']
        bb = detection[0]['facial_area']
        loc,emocoes = [],[]

        results = detector.detect_emotions(frame)
        for r in results:
            loc.append(r["box"])
            emocoes.append(r["emotions"])
        if len(bb)>0:
            coll = client.get_or_create_collection(name="celebridades",metadata={"hnsw:space": "cosine"})
            result = coll.query(query_embeddings=[imgEnc], n_results=10,include=["metadatas", "distances"] )
            nomes = [result['metadatas'][0][i]['nome'] for i in range(len(result['metadatas'][0])) ]
            nomes,classificacao = np.array(nomes),np.array(np.array(result['distances'])[0])

            _, seq = np.unique(nomes,return_index=True)
            nomes,classificacao = nomes[seq],classificacao[seq]
            classificacao =  softmax(classificacao)
            seq = np.argsort(classificacao)
            nomes,classificacao = nomes[seq],classificacao[seq]
            proximo,pct =result['metadatas'][0][0]['nome'],np.round(100*classificacao[0],0)
            ax1.clear()
            ax1.bar(nomes,classificacao)
            ax1.set_ylim(np.min(classificacao)-0.01,np.max(classificacao)+0.01)
            ax1.set_xticklabels(nomes, rotation=25, ha="right")
            fig.canvas.draw_idle()
            plt.pause(0.001)
            if len(loc)>0:
                dist = []
                for box in loc:
                    x, y, w, h = box
                    cx = x + w / 2
                    cy = y + h / 2
                    bx = bb["x"] + bb["w"] / 2
                    by = bb["y"] + bb["h"] / 2
                    d = (cx - bx)**2 + (cy - by)**2
                    dist.append(d)

                maisProximo = np.argmin(dist)
                e = emocoes[maisProximo]
                ax2.clear()
                angles = np.linspace(0,2*np.pi,len(e.values())+1)[:-1]
                ax2.set_ylim(0,1)
                ax2.bar(angles,e.values() )
                ax2.set_xticks(angles, ['raiva', 'nojo', 'medo', 'feliz', 'triste', 'surpresa', 'neutro'], color='grey', size=12)
                fig.canvas.draw_idle()
                plt.pause(0.001)

    if len(bb)>0:
        rec = f"{proximo} - {pct}%"
        cv2.rectangle(frame, (bb['x'], bb['y']), (bb['x']+bb['w'], bb['y']+bb['h']), (0, 255, 255), 2)
        text_size, _ = cv2.getTextSize(rec, fonte,escala,3)
        text_w, text_h = text_size
        cv2.rectangle(frame, (bb['x'], bb['y']), (bb['x']+text_w, bb['y']-text_h), (0, 255, 255), -1)
        cv2.putText(frame, rec, (bb['x'], bb['y']),fonte,escala, (0, 0, 0), 3, cv2.LINE_AA)
        
    cv2.imshow(nomeTela, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


