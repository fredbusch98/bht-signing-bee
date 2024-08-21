# Signing Bee

## Getting Started

Folgende python packages müssen installiert werden: `mediapipe`, `pillow`, `opencv-python`, `torch`
```bat
pip install mediapipe
pip install pillow
pip install opencv-python
pip install torch
```
`scikit-learn` (wird nur benötigt, wenn das Modell neu trainiert oder evaluiert werden soll)

```bat
pip install scikit-learn
```

## How to use
Es wird eine Webcam benötigt! Nach dem Start der Anwendung wird ein Wort aus einer vordefinierten Liste von Wörtern angezeigt. Diese müssen vor der Webcam in amerikanischer Zeichensprache (ASL) buchstabiert werden. Um Benutzern zu helfen, die nicht mit dem ASL-Alphabet vertraut sind, gibt es ein Hilfsbild in der Benutzeroberfläche, das die aktuell erforderliche Gebärde anzeigt. Für den Fall, dass ein Buchstabe mal nicht erkannt wird und man zum nächsten übergehen möchte, haben wir auch Tastaturunterstützung implementiert, so dass Buchstaben übersprungen werden können.

Navigiere zum Ordner `/src` und starte die Anwendung
```bat
cd src
python signing_bee.py
```

Application shortcuts
```
'z'   - Beendet die Anwendung
'j'   - Visualisierung der Hand Landmarks an- / ausschalten
'esc' - Vollbildmodus an- / ausschalten
```

## Exposé - Signing Bee
Frederik Busch, Marc Waclaw, Lennart Reinke

Unsere Idee ist es, ein kleines Buchstabierspiel (engl. spelling bee) zu entwickeln. Buchstabieren jedoch nicht im herkömmlichen Sinne mit der Stimme oder per Tastatureingabe, sondern mittels Sign Language.

Dem Nutzer wird ein zufälliges Wort angezeigt und zusätzlich eine Hilfestellung in Form eines Bildes, das die zugehörige Gebärde für den erwarteten Buchstaben zeigt. Mit Hilfe eines Convolutional Neural Network (CNN) erkennen wir dann die Handgeste des Nutzers und zeigen den Buchstabier-Fortschritt in der Benutzeroberfläche an. Ziel ist es, eine Applikation zu entwickeln, die es ermöglicht Sign Language zu lernen und im Zuge des Projektes herauszufinden, welche Gebärden von unserem Algorithmus gut erkannt werden und welche weniger gut.

Zum Trainieren unseres CNNs verwenden wir den *Sign Language MNIST* Datensatz. Dieser enthält etwa 35.000 28 x 28 Pixel Graustufenbilder von Gebärden des amerikanischen Sign Language Alphabets (27.455 Trainingsbilder, 7172 Testbilder). Der Datensatz schließt die Gebärden für die Buchstaben *J* und *Z* aus, weil diese im Gegensatz zu allen anderen Buchstaben Bewegung erfordern. [1] Eine sinnvolle Erkennung dieser Gebärden ist mit dem Datensatz nicht möglich. Deshalb werden auch wir in unserer Anwendung auf Wörter verzichten, die diese Buchstaben (*J*, *Z*) enthalten.

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

## Ergebnisse

Accuracy: 99,05%  
Precision: 99,00%  
Recall: 98,99%  
F1-Score: 98,98%

### Konfusionsmatrix

![Konfusionsmatrix](resources/results/confusion_matrix.jpg)

### ROC_AUC Kurve

![ROC_AUC Kurve](resources/results/roc_auc_curve.jpg)

## Download Links

Preprocessed Hand Landmark Data - https://drive.google.com/file/d/1EIZTrGiA6iRfyixYQjb6vpjJg2vGGb0p/view?usp=sharing

ASL(American Sign Language) Alphabet Dataset - https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?select=ASL_Alphabet_Dataset

ASL Alphabet - https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train
