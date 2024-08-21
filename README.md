# Signing Bee

## Getting Started

Install the following python packages: mediapipe, pillow, opencv-python, torch, scikit-learn (only needed if you want to train or evaluate the model)

```bat
pip install mediapipe
pip install pillow
pip install opencv-python
pip install torch
pip install scikit-learn
```

Navigate to the 'src' folder and start the application
```bat
cd src
python signing_bee.py
```

Application shortcuts
```
'z'   - quits the application
'j'   - toggles visualizing the hand-landmarks
'esc' - toggles full screen
```

## How to use
You need a webcam to use this software! After starting the application it shows a word from a pre set list of words. They need to be spelled out in front of the webcam using the American Sign Language. To help users that are not familiar with the ASL Alphabet there is a helper image present in the UI that shows the currently required gesture. If in some case a letter won't be recognized and you want to move on to the next one, we implemented keyboard support aswell, so you can skip letters.


## Exposé - Signing Bee
Frederik Busch, Marc Waclaw, Lennart Reinke

Unsere Idee ist es, ein kleines Buchstabierspiel (engl. spelling bee) zu entwickeln. Buchstabieren jedoch nicht im herkömmlichen Sinne mit der Stimme oder per Tastatureingabe, sondern mittels Sign Language.

Dem Nutzer wird ein zufälliges Wort angezeigt und zusätzlich eine Hilfestellung in Form eines Bildes, das die zugehörige Geste für den erwarteten Buchstaben zeigt. Mit Hilfe eines Convolutional Neural Network (CNN) erkennen wir dann die Handgeste des Nutzers und zeigen den Buchstabier-Fortschritt in der Benutzeroberfläche an. Ziel ist es, eine Applikation zu entwickeln, die es ermöglicht Sign Language zu lernen und im Zuge des Projektes herauszufinden, welche Gesten von unserem Algorithmus gut erkannt werden und welche weniger gut.

Zum Trainieren unseres CNNs verwenden wir den *Sign Language MNIST* Datensatz. Dieser enthält etwa 35.000 28 x 28 Pixel Graustufenbilder von Gesten des amerikanischen Sign Language Alphabets (27.455 Trainingsbilder, 7172 Testbilder). Der Datensatz schließt die Gesten für die Buchstaben *J* und *Z* aus, weil diese im Gegensatz zu allen anderen Buchstaben Bewegung erfordern. [1] Eine sinnvolle Erkennung dieser Gesten ist mit dem Datensatz nicht möglich. Deshalb werden auch wir in unserer Anwendung auf Wörter verzichten, die diese Buchstaben (*J*, *Z*) enthalten.

Wir werden das Projekt mit Hilfe von OpenCV, MediaPipe, PyTorch und tkinter umsetzen. OpenCV verwenden wir für die Bildverarbeitung, MediaPipe liefert nützliche Funktionen zur Hand-Detection, PyTorch nutzen wir zur Umsetzung unseres CNNs und tkinter für die Erstellung der Benutzeroberfläche. [2] [3] [4] [5]

Resources: 
* [1] Sign Language MNIST - https://www.kaggle.com/datasets/datamunge/sign-language-mnist
* [2] OpenCV - https://opencv.org/
* [3] MediaPipe - https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
* [4] PyTorch - https://pytorch.org/docs/stable/index.html 
* [5] tkinter - https://docs.python.org/3/library/tkinter.html

## Neuer Datensatz
Das Projektteam hat festgestellt, dass die Ergebnisse mit dem Sign Language MNIST-Datensatz nicht den Anforderungen entsprochen haben. Die Hauptprobleme lagen in der hohen Fehleranfälligkeit bei der Erkennung und der Abhängigkeit der Erkennungsqualität von Lichtverhältnissen und Hintergrundfarbe im Webcam-Frame. Diese Schwierigkeiten können auf die Verwendung von globalen Deskriptoren, in Form von 28x28 Graustufenbildern, zurückgeführt werden. Diese Bilder bieten nur eine begrenzte Menge an Informationen für die Erkennung und unterscheiden sich möglicherweise nicht ausreichend bei variierenden Bedingungen.

Aufgrund dieser Einschränkungen haben wir uns entschieden, einen neuen Datensatz zu verwenden und das Feature von MediaPipe zur Erkennung von Hand-Landmarks einzuführen. Bei diesem Ansatz verwenden wir 21 Hand-Landmarks pro Hand, wobei jeder Landmark durch normalisierte X-, Y- und Z-Koordinaten beschrieben wird. Dieser Wechsel ermöglicht es uns, von den globalen Deskriptoren der 28x28 Graustufenbilder auf detailliertere lokale Features umzusteigen.

**Mathematische Erklärung der lokalen Features:**

Für jede Hand werden 21 Hand-Landmarks erkannt. Jeder dieser Landmarks besteht aus 3 Koordinaten (X, Y, Z). Daher können wir die Anzahl der lokalen Features wie folgt berechnen:

- Anzahl der Hand-Landmarks pro Hand: 21
- Anzahl der Koordinaten pro Landmark: 3

Die Gesamtanzahl der lokalen Features pro Hand ist daher:

\[ \text{Gesamtanzahl der lokalen Features} = \text{Anzahl der Hand-Landmarks} \times \text{Anzahl der Koordinaten pro Landmark} \]
\[ \text{Gesamtanzahl der lokalen Features} = 21 \times 3 = 63 \]

Also besteht jede Handrepräsentation aus 63 lokalen Features (Koordinaten), die verwendet werden können, um die Handgesten zu erkennen. Dies bietet eine viel detailliertere und spezifischere Beschreibung der Handposition und -bewegung im Vergleich zu den ursprünglichen globalen Deskriptoren.

Für die Entwicklung und das Training unseres Modells haben wir zwei Datensätze verwendet:

- **Trainingsdatensatz**: [ASL (American Sign Language) Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?select=ASL_Alphabet_Dataset), der etwa 9000 Bilder pro Buchstabe enthält.
- **Testdatensatz**: [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train), der etwa 3000 Bilder pro Buchstabe umfasst.

Da beide Datensätze ausschließlich Bilddaten enthalten, wir jedoch Hand-Landmarks zur Erkennung verwenden möchten, mussten die Daten zunächst vorverarbeitet werden. Hierzu haben wir das Skript `data_preprocessor.py` entwickelt. Dieses Skript liest alle Bilder aus den Datensätzen ein und nutzt MediaPipe zur Identifikation der Hand-Landmarks. Die erkannten Landmarks werden anschließend in eine CSV-Datei geschrieben, wobei das zugehörige Label in der ersten Spalte als Präfix angegeben wird. Da nicht bei jedem Bild Hand-Landmarks erkannt werden konnten, reduzierte sich die Datenmenge im Vorverarbeitungsprozess. Der finale Trainingsdatensatz enthält nun im Durchschnitt 5728 Datenpunkte pro Buchstabe, und der finale Testdatensatz weist im Durchschnitt 2360 Datenpunkte pro Buchstabe auf. Diese Menge übertrifft deutlich die Anzahl der Datenpunkte im Sign Language MNIST-Datensatz, der im Durchschnitt nur 1145 Datenpunkte pro Buchstabe bereitstellte.

Um sicherzustellen, dass die Trainingsdaten gleichmäßig verteilt sind, implementierten wir das Skript `label_counter.py`, das die Gesamtanzahl der Datenpunkte pro Label ermittelt. Es stellte sich heraus, dass die Buchstaben M und N nach der Vorverarbeitung signifikant weniger Datenpunkte aufwiesen als der Durchschnitt, mit lediglich 3017 (M) bzw. 3432 (N) Datenpunkten. Zur Balance des Trainingsdatensatzes wurden diese Buchstaben durch Duplizieren von Datenpunkten aufgestockt, um sicherzustellen, dass für jeden Buchstaben mindestens 5000 Datenpunkte vorhanden sind.


## Download Links

Preprocessed Hand Landmark Data - https://drive.google.com/file/d/1EIZTrGiA6iRfyixYQjb6vpjJg2vGGb0p/view?usp=sharing

ASL(American Sign Language) Alphabet Dataset - https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?select=ASL_Alphabet_Dataset

ASL Alphabet - https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train
