# Signing Bee

## Getting Started

Install the following python packages: mediapipe, pillow, opencv-python, torch

```bat
pip install mediapipe
pip install pillow
pip install opencv-python
pip install torch
```

### Exposé - Signing Bee
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
