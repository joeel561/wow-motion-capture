# WoW Motion Capture – Anleitung

Steuert deinen Charakter mit Körperposen über die Webcam. **Nur der Körper zählt**
(Schultern, Arme, Hände, Hüfte, Beine, Füße) – Kopf und Gesicht werden ignoriert,
du kannst also frei auf den Bildschirm schauen.

## Starten

Doppelklick auf **`start.bat`**. Beim ersten Start wird die Python-Umgebung
eingerichtet (einige Minuten) und das Pose-Modell geladen. Das schwarze
Konsolenfenster offen lassen – dort stehen eventuelle Fehlermeldungen.

## Posen kalibrieren

1. **Ruhepose aufnehmen** – so, wie du normal vor der Kamera stehst/sitzt.
   Sie bedeutet „nichts tun“ und gilt für alle Bereiche. Unbedingt zuerst!
2. **Neue Pose:** Name, Bereich, Taste und Modus eintragen → **● Neue Pose aufnehmen**.
   Countdown (Piepser) → Pose einnehmen → 3 s halten (langer Piep = Start,
   Doppelpiep = fertig). Bewege dich dabei leicht innerhalb der Pose.
3. Erkennt er eine Pose schlecht: Pose auswählen → **Samples hinzufügen**
   (gern aus etwas anderer Position/Entfernung).

### Bereiche – mehrere Posen gleichzeitig

Jeder Bereich wird unabhängig erkannt. So kannst du z. B. gleichzeitig nach vorn
lehnen (laufen) und mit dem Arm einen Zauber auslösen.

| Bereich | wertet aus | gut für |
|---|---|---|
| Rumpf (Neigen) | Schultern relativ zur Hüfte | Vor-/Zurück-/Seitlich lehnen → Laufen, Drehen |
| Arme & Hände | Arme relativ zu den Schultern | Zauber, Angriffe, Aktionsleiste |
| Beine & Füße | Beine relativ zur Hüfte | Knie hoch → Springen, Ausfallschritt |
| Ganzer Körper | alles | große Ganzkörper-Gesten |

Tipp: Wer sitzt, nutzt nur *Rumpf* und *Arme* – Beine werden automatisch
ignoriert, wenn sie nicht im Bild sind.

### Modus

- **Halten** – Taste bleibt gedrückt, solange du die Pose hältst (Laufen: `w`, Drehen: `a`/`d`).
- **Antippen** – einmal drücken beim Einnehmen der Pose (Zauber, Springen). Für die
  nächste Auslösung kurz in die Ruhepose zurück.

Tasten: `w`, `1`, `space`, `shift+2`, `ctrl+1`, `f1`, `tab`, `num5`,
`links/rechts/hoch/runter`, Maus: `lmb`, `rmb`, `mmb`, `maus4`, `maus5`.

## Kamera-Felder

Zusätzlich zu Posen kannst du **Rechtecke im Kamerabild** festlegen. Sobald das
gewählte Körperteil im Rechteck ist, wird die Taste ausgelöst.

1. Tab **Kamera-Felder** → **▭ Rechteck im Kamerabild aufziehen**.
2. Im Kamerabild mit gedrückter linker Maustaste ein Rechteck ziehen (weiß = Vorschau).
3. Name, **Körperteil**, Taste und Modus eintragen → **Neues Feld speichern**.

Körperteile: *Eine Hand*, *Rechte Hand*, *Linke Hand*, *Ein Fuß*, *Rechter Fuß*,
*Linker Fuß*, *Körpermitte* (gut für „zur Seite treten“), *Irgendein Körperteil*.

Im Bild: gelber Rahmen = bereit, grün gefüllt = gerade aktiv. Das Bild ist
gespiegelt wie ein Spiegel – greifst du nach rechts, geht die Hand auch im Bild nach rechts.
Felder laufen gleichzeitig mit den Posen. Feld verschieben: in der Liste auswählen,
neues Rechteck aufziehen, **Änderungen übernehmen**.

Ideen: Hand oben rechts = Zauber `1`, Hand oben links = Heilung `2`,
Körpermitte im linken Feld = Ausweichen nach links `q` (Halten),
Fuß im unteren Feld = Springen `space`.

## Beispiel-Setup

| Pose | Bereich | Taste | Modus |
|---|---|---|---|
| Ruhepose | – | – | – |
| Vorlehnen | Rumpf | `w` | Halten |
| Zurücklehnen | Rumpf | `s` | Halten |
| Links lehnen | Rumpf | `a` | Halten |
| Rechts lehnen | Rumpf | `d` | Halten |
| Rechter Arm nach vorn | Arme | `1` | Antippen |
| Linker Arm zur Seite | Arme | `2` | Antippen |
| Beide Arme hoch | Arme | `3` | Antippen |
| Rechtes Knie hoch | Beine | `space` | Antippen |
| Arme vor der Brust kreuzen | Arme | `tab` | Antippen |

## Im Spiel

- **F12** (einstellbar) schaltet die Steuerung ein/aus – auch wenn das Spiel im
  Vordergrund ist. Hoher Doppelpiep = an, tiefer = aus.
- Tasten gehen **nur an ein Fenster, dessen Titel „World of Warcraft“ enthält**.
  Heißt dein Spielfenster anders: Unter *Status* steht „Zuletzt aktives Fenster: …“ –
  einen Teil davon unter *Einstellungen → Spielfenster-Titel* eintragen.
- Spiel am besten im **Fenstermodus (Vollbild)** laufen lassen.
- Beim Ausschalten, Fensterwechsel oder Schließen werden alle gehaltenen Tasten losgelassen.

## Feintuning

- Unter dem Kamerabild steht pro Bereich, wie nah du jeder Pose bist (kleiner = näher).
- **Toleranz**: Wird eine Pose nicht erkannt → höher; löst sie zu leicht aus → niedriger.
- **Stabilität**: mehr Frames = weniger Fehlauslöser, aber etwas mehr Verzögerung.
- Posen sollten sich deutlich unterscheiden. Gleichmäßiges Licht hilft.

## Probleme

- **Keine Kamera:** Einstellungen → Kamera-Nr. ändern → *Anwenden*
  (das Tool sucht auch automatisch). Andere Programme, die die Kamera nutzen, schließen.
- **„Anwendungssteuerungsrichtlinie hat diese Datei blockiert“:** Windows Smart App
  Control blockiert neuere MediaPipe-Versionen. Deshalb ist Version 0.10.21 fest eingestellt –
  `requirements.txt` nicht ändern.
- Gespeichert wird alles in `posen.json` (Posen, Kamera-Felder, Einstellungen) neben dem Skript.
- Läuft das Tool zweimal, bekommt nur eine Instanz die Kamera („Keine Kamera gefunden“).

## Hinweis

Jede Pose löst genau eine Taste aus, wie eine alternative Tastatur. Das Tool
automatisiert nichts selbst. Ob Bewegungssteuerung auf deinem Server/Spiel erlaubt
ist, regeln die dortigen Nutzungsbedingungen – im Zweifel nachfragen.
