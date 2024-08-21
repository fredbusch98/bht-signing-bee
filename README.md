# Signing Bee

## Getting Started

Install the following python packages: mediapipe, pillow, opencv-python, torch

```bat
pip install mediapipe
pip install pillow
pip install opencv-python
pip install torch
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

## How to use (temporarily until CNN is implemented):
You need a Webcam to use this software! After starting the application shows a list of words. They need to be spelled out in front of the webcam using the American Sign Language. To help users that are not familiar with the ASL Alphabet there is a helper image present in the UI that shows the required gesture.


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
Im Zuge der Entwicklung des Projektes unter Verwendung des zuvor genannten Datensatzes sind einige Probleme aufgetreten. Die Ergebnisse die wir mit dem Sign Language MNIST erzielen konnten waren für unsere Ansprüche nicht genügend. Es gab viele Verwechslungen bei der Erkennung und die Qualität der Erkennung war extrem abhängig von Lichtverhältnissen, so wie der Hintergrundfarbe im Webcam-Frame. Diese Probleme kamen höchstwahrscheinlich dadurch zustande, das beim Sign Language MNIST rudimentäre globale Deskriptoren (28x28 Graustufen Bilder) zum Training des Models und zur späteren Erkennung der Gesten verwendet werden. Aus diesem Grund haben wir uns schlussendlich dazu entschieden einen neuen Datensatz zu verwenden und außerdem das Feature von MediaPipe zuer Erkennung von sogenannten Hand Landmarks für das Training und die Erkennung zu verwenden. Pro Hand gibt es 21 Hand Landmarks die jweils aus normalisierten X, Y und Z Koordinaten bestehen.

Trainingsdatensatz - [ASL(American Sign Language) Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?select=ASL_Alphabet_Dataset)
Dieser Datensatz enthält etwa 9000 Bilder pro Buchstabe. 

Testdatensatz - [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train)
Dieser Datensatz enthält etwa 3000 Bilder pro Buchstabe.

Da die beiden Datensätze ausschließlich Bilddatenpunkte enthalten, wir jedoch die Hand Landmarks nutzen wollen mussten wir die Daten noch vorverarbeiten. Hierfür ist das Skript data_preprocessor.py zuständig. In diesem Skript werden alle Bilder aus dem Datensatz eingelesen, dann wird versucht mit Hilfe von MediaPipe die Hand Landmarks in dem Bild zu erkennen und, wenn welche erkannt wurden, werden diese in eine .csv Datei geschrieben, jeweils mit dem zugehörigen Label als Prefix in der ersten Spalte. Da hier nicht bei jedem Bild Hand Landmarks erkannt wurden, hat sich die Datenmenge in diesem Vorverarbeitungsschritt reduziert, so dass wir im finalen Trainingsdatensatz nur noch 5728 und im finalen Testdatensatz nur noch 2360 Datenpunkte im Durchscnitt pro Buchstabe hatten. Das ist aber immer noch weit aus mehr als im Sign Language MNIST. Hier waren es bei den Trainingsdaten nur durchscnittlich 1145 Datenpunkte pro Buchstabe. Um anschließend noch sicherzustellen, dass die Trainingsdaten gleichgewichtig verteilt waren haben wir das Skript label_counter.py implementiert. Hier bekommen wir zu jedem Label die Gesamtmenge an Datenpunkte ausgegeben. Bei den Trainingsdaten hat sich herausgestellt, das M und N hier nach dem Preprocessing deutlich weniger Datenpunkte hatten, als der Durchschnitt, mit gerade mal 3017 (M) und 3432 (N). Um nun unseren Trainingsdatensatz auszubalancieren haben wir hier Datenpunkte dupliziert, so dass auch hier mindestens 5000 Datenpunkte vorhanden waren. 


## Download Links

Preprocessed Hand Landmark Data - https://drive.google.com/file/d/1EIZTrGiA6iRfyixYQjb6vpjJg2vGGb0p/view?usp=sharing

ASL(American Sign Language) Alphabet Dataset - https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset?select=ASL_Alphabet_Dataset

ASL Alphabet - https://www.kaggle.com/datasets/grassknoted/asl-alphabet?select=asl_alphabet_train
