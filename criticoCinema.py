from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import face_recognition
import chromadb
import cv2
import uuid
import numpy as np
from fer.fer import FER


def softmax(distances):
    inverted = -np.array(distances)
    exp_vals = np.exp(inverted - np.max(inverted))
    return exp_vals / np.sum(exp_vals)


client = chromadb.PersistentClient(path="./encodingCelebridades")
coll = client.get_or_create_collection(name="celebridades")

fonte, escala = cv2.FONT_HERSHEY_SIMPLEX, 0.5
cascade_path = 'haarcascade_frontalface_default.xml'
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
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Falha ao ler a imagem.")
        break

        
    bb = face_recognition.face_locations(frame)
    results = detector.detect_emotions(frame)
    loc,emocoes = [],[]
    for r in results:
        loc.append(r["box"])
        emocoes.append(r["emotions"])
    
    

    if len(bb)>0:
        bb = bb[0]
        imgEnc = face_recognition.face_encodings(frame)[0]
        coll = client.get_or_create_collection(name="celebridades",metadata={"hnsw:space": "cosine"})
        result = coll.query(query_embeddings=[imgEnc], n_results=100,include=["metadatas", "distances"] )
        nomes = [result['metadatas'][0][i]['nome'] for i in range(len(result['metadatas'][0])) ]
        nomes,classificacao = np.array(nomes),np.array(np.array(result['distances'])[0])
        
        _, seq = np.unique(nomes,return_index=True)
        nomes,classificacao = nomes[seq],classificacao[seq]
        classificacao =  softmax(classificacao)
        seq = np.argsort(classificacao)
        nomes,classificacao = nomes[seq],classificacao[seq]

        rec = f"{result['metadatas'][0][0]['nome']} - {np.round(100*classificacao[0],0)}%"
        
        if len(loc)>0:
            maisProximo = np.argmin(np.array(loc) -np.array(bb)**2,axis=0)[0]
            e = emocoes[maisProximo]
            ax2.clear()
            angles = np.linspace(0,2*np.pi,len(e.values())+1)[:-1]
            ax2.set_ylim(0,1)
            ax2.bar(angles,e.values() )
            ax2.set_xticks(angles, ['raiva', 'nojo', 'medo', 'feliz', 'triste', 'surpresa', 'neutro'], color='grey', size=12)
            
            fig.canvas.draw_idle()
            plt.pause(0.001)
        

        cv2.rectangle(frame, (bb[3], bb[2]), (bb[1], bb[0]), (0, 255, 255), 2)
        text_size, _ = cv2.getTextSize(rec, fonte,escala,3)
        text_w, text_h = text_size
        cv2.rectangle(frame, (bb[3], bb[0]-2*text_h), (bb[3]+text_w, bb[0]), (0, 255, 255), -1)
        cv2.putText(frame, rec, (bb[3], bb[0]-10),fonte,escala, (0, 0, 0), 3, cv2.LINE_AA)
   
        ax1.clear()
        ax1.bar(nomes,classificacao)
        ax1.set_ylim(np.min(classificacao)-0.01,np.max(classificacao)+0.01)
        ax1.set_xticklabels(nomes, rotation=25, ha="right")
        
        fig.canvas.draw_idle()
        plt.pause(0.001)
        
        

    cv2.imshow(nomeTela, frame)

    # 4. Stop the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


