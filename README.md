# WoW Motion Capture

Steuere deinen Spielcharakter mit deinem Körper – nur über eine normale Webcam.
Du kalibrierst eigene Posen und Kamera-Felder, das Tool übersetzt sie in
Tastendrücke fürs Spiel.

- **Nur der Körper zählt:** Schultern, Arme, Hände, Hüfte, Beine, Füße.
  Kopf und Gesicht werden komplett ignoriert.
- **Eigene Posen kalibrieren:** Name und Taste eingeben, Pose 3 Sekunden halten – fertig.
- **Mehrere Bereiche gleichzeitig:** Rumpf, Arme und Beine werden getrennt erkannt
  (z. B. nach vorn lehnen = laufen *und* Arm heben = Zauber).
- **Kamera-Felder:** Rechtecke im Kamerabild aufziehen – kommt eine Hand, ein Fuß
  oder die Körpermitte hinein, wird eine Taste ausgelöst.
- **Halten oder Antippen** pro Pose/Feld, Tastenkombinationen (`shift+2`) und Maustasten.
- **Sicher:** globale Ein/Aus-Taste (F12), Tasten gehen nur ans Spielfenster,
  beim Ausschalten werden alle Tasten losgelassen.

## Voraussetzungen

- Windows 10/11
- Webcam
- [Python 3.12](https://www.python.org/downloads/) – oder per
  `winget install --id Python.Python.3.12 --source winget --scope user`

## Start

Repository herunterladen und **`start.bat`** doppelklicken. Beim ersten Start
werden die Abhängigkeiten installiert und das Pose-Modell geladen (einige Minuten).

Bedienung, Beispiel-Setup und Tipps: siehe [ANLEITUNG.md](ANLEITUNG.md).

## Technik

- Körpererkennung: [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
  (läuft lokal, das Kamerabild verlässt deinen Rechner nicht)
- Posenerkennung: k-nächste-Nachbarn auf normierten Körperpunkten
- Tastendrücke: Windows `SendInput` mit Scancodes
- Oberfläche: Tkinter

`mediapipe` ist auf Version 0.10.21 festgelegt, weil Windows Smart App Control
neuere Versionen blockiert.

Deine Kalibrierung wird lokal in `posen.json` gespeichert (per `.gitignore`
vom Repository ausgeschlossen).

## Hinweis

Inoffizielles Fan-Projekt, nicht mit Blizzard Entertainment verbunden.
World of Warcraft ist eine Marke von Blizzard Entertainment, Inc.
Jede Pose bzw. jedes Feld löst genau eine Taste aus – wie eine alternative
Tastatur. Ob Bewegungssteuerung erlaubt ist, regeln die Nutzungsbedingungen
des jeweiligen Spiels bzw. Servers.

## Lizenz

[MIT](LICENSE)
