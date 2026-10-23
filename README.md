# Wikipedia Timecurve

Das Projekt visualisiert die Entwicklung eines Wikipedia-Artikels als
zweidimensionale semantische Landschaft. Jeder Punkt ist eine vollständige
Artikelrevision, jede Linie verbindet zwei chronologisch aufeinanderfolgende
Revisionen. Labels werden ohne feste Zoomstufen kartografisch verteilt: Sobald
beim Zoomen oder Verschieben genügend freier Bildschirmraum entsteht, erscheint
dort der nächste priorisierte Begriff.

## Finale Palestine-Version

Die Präsentationsdatei ist
[`output/Palestine_final.html`](output/Palestine_final.html).

Der finale Datensatz umfasst:

- Artikel: `Palestine`
- Zeitraum: 2. Dezember 2012 bis 23. Juli 2014
- 620 vollständige Revisionen und damit 619 Kanten
- 619 von 619 Kantenlabels aus vollständigen Revisionsvergleichen
- Labelmodell: `gpt-5.6-terra`
- Embeddingmodell: `sentence-transformers/all-MiniLM-L6-v2`
- Projektion: klassische metrische MDS
- Clustering: HDBSCAN auf der sichtbaren 2D-Projektion
- 44 Cluster und 140 Noise-Punkte
- 71 exakte SHA1-Reverts
- deterministische Punktklassifikation: 87 nur hinzugefügt, 63 nur entfernt,
  288 gemischt und 181 ohne bereinigte Textänderung; der erste Punkt ist der Start

## Methode

```text
Wikipedia-XML
→ vollständige bereinigte Revisionen
→ höchstens 256 Tokens lange Encoding-Chunks
→ gemittelter und L2-normalisierter Vektor je Revision
→ Kosinus-Distanzmatrix
→ klassische MDS in zwei Dimensionen
→ HDBSCAN auf den MDS-Koordinaten
→ chronologische Revisionskurve
→ Terra-Volltextvergleich für jede Kante
→ kontinuierliches kartografisches Labeling aus einer fünfteiligen Hierarchie
```

Die Distanz zweier Revisionen ist:

```text
1 - dot(L2-normalisierter Revisionsvektor A,
        L2-normalisierter Revisionsvektor B)
```

## Kantenlabels

Für jede der 619 Kanten erhält `gpt-5.6-terra` die beiden vollständigen
bereinigten Revisionstexte vor und nach der Änderung sowie Autor, Zeitstempel
und – sofern im XML vorhanden – den Bearbeitungskommentar beider Revisionen.
Ein vorberechneter Diff wird nicht
übergeben; das Modell analysiert die beiden Volltexte direkt. Es erzeugt ein
unteilbares Label mit zwei bis drei Wörtern, das bereits Objekt und
Handlung beziehungsweise Ergebnis der konkreten Änderung nennt, sowie eine
ausführliche Beschreibung der erkannten Änderung. Dadurch wird eine Kante beim
Annähern inhaltlich genauer und wiederholt nicht nur ihr übergeordnetes Thema.
Die Metadaten sind lediglich zusätzlicher Kontext; der tatsächliche
Textunterschied bleibt maßgeblich. Wort-Diffs und Satzfenster werden nicht
erzeugt oder gesendet.

Die Modellantworten werden nach jeder Kante gecacht. Ein abgebrochener Lauf
kann deshalb ohne erneute Aufrufe fortgesetzt werden. Beim späteren Zoomen
finden keine API-Aufrufe statt.

Bytegleiche bereinigte Revisionstexte werden deterministisch als
`No Content Change` beschrieben und nicht an das Modell gesendet. Umgekehrt
wird eine neutrale Modellantwort verworfen, sobald sich die beiden bereinigten
Volltexte unterscheiden. Dadurch können Kurzlabel und Detailbeschreibung nicht
mehr widersprüchlich behaupten, dass eine inhaltliche Änderung zugleich
unverändert sei.

Die übergeordneten Ebenen werden vor dem Rendern mit derselben OpenAI-API aus
den gespeicherten Kantenlabels und Keywords ihres jeweiligen Bereichs erzeugt.
Terra formuliert für jeden zeitlichen Abschnitt, Cluster und jedes Makrothema
genau einen Überbegriff. Die Antworten werden im Hierarchie-JSON gecacht.

## Kartografische Hierarchie

```text
Artikel
└── Makrothema
    └── Themencluster
        └── zeitlicher Abschnitt
            └── Revisionskante
```

Die Hierarchie liefert Kandidaten und kartografische Prioritäten. Artikel-,
Makro-, Cluster-, Abschnitts- und kurze Kantenlabels werden bei jedem Maßstab
gemeinsam in Bildschirmkoordinaten geprüft. Ihre Reihenfolge ergibt sich aus
der zur Zoomstufe passenden semantischen Tiefe, der inhaltlichen Wichtigkeit
und einem Stabilitätsbonus für bereits sichtbare Labels. Es gibt kein globales
Anzahl- oder Bildschirmflächenbudget: Nach jedem höher priorisierten Kandidaten
werden auch alle untergeordneten Kandidaten geprüft. Ein Label wird angezeigt,
sobald seine eigene gemessene Bildschirm-Textbox vollständig im Sichtfenster
liegt, mit keiner bereits platzierten Textbox kollidiert und mindestens einen
der von ihm beschriebenen Revisionspunkte überdeckt.
Makrothemen besitzen keine eigenen Flächen, Konturen oder Farbcodierung;
sichtbar bleiben ausschließlich ihre direkten Labels. Die zugehörigen
Clusterhüllen werden neutral in der Trajektorienfarbe dargestellt.
Die redaktionell festgelegte Position eines übergeordneten Labels bleibt sein
semantisches Zentrum. Makro-, Cluster- und Abschnittslabels werden dort
zentriert und können anschließend zu bis zu acht zentral gelegenen
Revisionspunkten ihres eigenen Datengebiets ausweichen; der Artikelname bleibt
fest im Zentrum. Für
jedes Kantenlabel werden bis zu zehn Kandidaten an fünf Stellen und auf beiden
Seiten des aktuell sichtbaren Bézier-Abschnitts geprüft. Die jeweils freiste
zulässige Stelle wird gespeichert und bei kleinen Ausschnittsänderungen
bevorzugt beibehalten. Eine alternative Position für ein Hierarchielabel ist
nur zulässig, wenn seine Textbox weiterhin über den Revisionspunkten des
eigenen Makrothemas, Themenclusters oder zeitlichen Abschnitts liegt. Freier
Raum außerhalb dieses Datengebiets ist kein gültiger Ausweichplatz.
Alle Labels behalten eine konstante Schriftgröße.
Die Größe kennzeichnet dabei die Hierarchiestufe: Der Artikelname erscheint
mit 28 px, Makrothemen mit 24 px, Themencluster mit 18 px, zeitliche Abschnitte
mit 13 px und Kantenlabels mit 10 px. Start und Ende bleiben eigenständige
Sonderlabels mit 14 px.
Die Zoomstufe bestimmt keine harte Sichtbarkeitsgrenze mehr, sondern eine
bevorzugte semantische Tiefe. In der Übersicht werden Artikel, Makrothemen und
Themencluster zuerst geprüft. Mit zunehmendem Zoom wandert die Priorität
kontinuierlich zu Themenclustern, zeitlichen Abschnitten, kurzen Kantenlabels
und schließlich zu ausführlichen Kantenbeschreibungen. Untergeordnete Labels
bleiben auf jeder normalen Zoomstufe Kandidaten und erscheinen zusätzlich,
wenn ihre tatsächliche Bildschirm-Textbox in einen freien Bereich passt.
Die jeweils nächstfeinere Ebene erhält nur einen moderaten Vorwärtsmalus und
kann dadurch bereits kurz vor ihrer idealen Zieltiefe freie Flächen nutzen.
Start und Ende bleiben hoch priorisierte Sondermarkierungen.
Jede sichtbare Kindebene soll mindestens einen neuen inhaltlichen Begriff oder
eine konkrete Änderungsrichtung hinzufügen. Kantenkurzlabels nennen deshalb
Gegenstand plus Handlung beziehungsweise Ergebnis; ihre ausführliche Variante
ergänzt Autor, Zeitpunkt und vollständigen Umfang. Eine Prüffunktion kann
inhaltlich unspezifische Kindlabels für die redaktionelle Kontrolle melden;
der normale Kartenaufbau bricht deswegen derzeit nicht automatisch ab.
Berühren sich zwei Textflächen, verschwindet das Label mit niedrigerer
Priorität. Zusätzlich rastert ein 8-px-Dichteindex die sichtbaren
Catmull–Rom-Kanten und Revisionspunkte. Für Hierarchielabels dient diese Dichte
nur als Auswahlkriterium zwischen Positionen, die das eigene Datengebiet
überdecken; Kantenlabels wählen unter ihren Kandidaten die Position mit der
geringsten Kanten- und Knotendichte. Unterschiedliche Ein-
und Ausblendeschwellen erzeugen Hysterese: Ein sichtbares Label darf bei einer
kleinen Verschlechterung bestehen bleiben, während ein neues Label eine
strengere Freiraumgrenze erfüllen muss. Die Platzierung wird erst nach dem
Ende einer Zoom- oder Pan-Geste neu berechnet und mit 150 ms überblendet.
Ein 8-px-Innenrand und SVG-Clipping halten Text vollständig im Plotbereich und
verhindern Überlagerungen mit Titelzeile oder Legende.
Textgleiche übergeordnete Artikel-, Makro-, Cluster- und Abschnittslabels
werden vor dem Rendern zusammengeführt. Räumlich getrennte Revisionskanten
behalten dagegen ihre eigenen Beschriftungen, auch wenn deren Texte identisch
sind. Analog zum Symbolabstand einer Straßenkarte werden gleichlautende
Kantenlabels nur unterdrückt, wenn ihre gewählten Bildschirmpositionen weniger
als 120 px auseinanderliegen. Nur vollständig überlagerte Kanten erhalten an
derselben Geometrie insgesamt ein Label. Dabei gewinnt zuerst eine vorhandene
ausführliche Erklärung, danach die größere Wortänderung, der Informationsumfang, die
Labelkonfidenz und der semantische Sprung. Die übrigen Kanten bleiben als
Geometrie und Revisionsereignisse erhalten.
Jedes sichtbare Label besitzt eine vollständig deckende, einheitlich dunkle
Textfüllung. Hierarchielabels verwenden einen weißen 4-px-Halo,
Kantenbeschriftungen einen weißen 2-px-Halo. Ausgeblendete
Kollisionskandidaten werden mit einer kurzen
Opacity-Überblendung entfernt.
Jede eigenständige Revisionskante bleibt ebenfalls ein Labelkandidat. Es gibt
keine globale Anzahl- oder Flächenobergrenze. Eine ausführliche mehrzeilige
Beschreibung wird nur versucht, wenn der sichtbare Spline-Abschnitt mindestens
140 Bildschirm-Pixel lang ist und ihre gemessene Textfläche sowohl
labelkollisionsfrei als auch weitgehend frei von Kanten und Knoten ist.
Andernfalls wird an der besten der übrigen Bézier-Positionen das kurze
Kartenlabel versucht. Erst wenn auch dieses keine ausreichend freie Position
findet, wird die Kantenbeschriftung ausgeblendet. Der vollständige Text bleibt
unabhängig davon als Hover-Tooltip erreichbar.
Hierarchie- und Kantenbeschriftungen werden gemeinsam nach einer zoomabhängigen
Priorität platziert. Dadurch reservieren beim Herauszoomen grobe Themenebenen
den Raum zuerst; beim Hineinzoomen werden die inhaltlich wichtigen
Kantenbeschriftungen schrittweise vorrangig. Bereits sichtbare Labels erhalten
einen Stabilitätsbonus, bleiben aber weiterhin an Kollisions- und
Freiraumprüfung gebunden. Eine vollständige Beschreibung wird erst ab
Zoomstufe 4,25 zugelassen. Die Freigabe hängt nicht davon ab, ob ein übergeordnetes
Label oder das Kurzlabel im vorherigen Bild sichtbar war. Die vollständigen
Beschreibungen werden unabhängig von den kurzen
Kartenlabels eingebunden, sodass redaktionell gewählte Kurzlabels erhalten
bleiben. Wo genügend Platz vorhanden ist, erscheint die vollständige
Beschreibung; andernfalls wird das Kurzlabel versucht oder die Beschriftung
bis zu einem stärkeren Zoom ausgeblendet. Effektiv punktförmige Kanten
verwenden ausschließlich das Kurzlabel. Die ausführliche Beschreibung bleibt
in den eingebetteten Daten und im Hover-Tooltip erhalten.

Die Füllfarbe jedes Revisionspunkts beschreibt deterministisch die Änderung
gegenüber seinem direkten Vorgänger: `added` (Goldgelb `#F2C14E`) bedeutet nur
hinzugefügte Wörter, `removed` (Violett `#8F5BD7`) nur entfernte Wörter,
`mixed` (Orange `#D9822B`) gleichzeitig hinzugefügte und entfernte Wörter und
`no_change` (Grau `#B8B8B8`) keine Änderung im bereinigten Text. Der erste
Punkt hat keinen Vorgänger und wird als Start in Dunkelgrau (`#343A40`)
gezeichnet. Die Zuordnung stammt ausschließlich aus dem Git-Wort-Diff; dafür
gibt es keinen LLM-Aufruf. Normale Revisionspunkte werden ohne zusätzliche
Kontur beziehungsweise Ring und mit hoher Farbdeckkraft gezeichnet.

Clusterhüllen, gruppeninterne Trajektorien und normale Richtungspfeile werden
neutral in der Trajektorienfarbe dargestellt. Zusätzliche Ringe für zeitliche
Abschnittspunkte werden nicht gezeichnet. Die
Makrothemen bleiben als semantische Labels erhalten, besitzen aber keine eigene
Farbcodierung und keinen Legendeneintrag.
Signalrot kennzeichnet exakte SHA1-Reverts.
Noise-Revisionen bleiben wie alle anderen Revisionen kreisförmig und erhalten
dieselbe Änderungsfarbcodierung.
Die konkrete Clusteridentität wird weiterhin durch die direkten semantischen
Clusterlabels vermittelt. Die Farblegende enthält keine Makrothemen mehr.
Jede Clusterhülle umfasst sämtliche projizierten Revisionspunkte des jeweiligen
HDBSCAN-Clusters. Für reguläre zweidimensionale Punktmengen wird eine leicht
erweiterte konvexe Hülle gezeichnet. Fallen die Punkte in der Projektion auf
eine Linie oder praktisch dieselbe Position zusammen, verbindet stattdessen
ein dünner Strich ihre unterscheidbaren Positionen. Zusätzliche
runde Endkappen werden nicht gezeichnet.

Die chronologische Haupttrajektorie wird als offene zentripetale
Catmull–Rom-Spline (`alpha = 0.5`) gerendert. Jede Verbindung zweier
aufeinanderfolgender Revisionen ist ein eigener kubischer Bézier-Abschnitt;
dadurch bleiben Kantenmetadaten und Breite einzeln ansprechbar.
Die Kurve interpoliert alle Revisionspunkte, verändert also weder deren
Position noch die Reihenfolge. Reflektierte Nachbarpunkte definieren die
beiden Endtangenten. Bei deckungsgleichen Revisionspunkten bleibt das Segment
punktförmig und begrenzte Tangenten verhindern visuelle Spitzen. Revert-Bögen
bleiben ein unabhängiges Overlay.

Punkte,
Abschnittsringe und Trajektorienlinien sind für die Detailansicht bewusst
kräftig dimensioniert. Auf jedem mindestens 34 Bildschirm-Pixel langen,
normalen Spline-Abschnitt zeigt ein Pfeil an der sichtbaren Kurvenmitte die
chronologische Revisionsrichtung. Der Pfeilwinkel folgt der lokalen
Spline-Tangente und behält beim Zoomen seine Bildschirmgröße.
Die festen Farben für Trajektorie und exakten Revert werden aus denselben
Farbkonstanten wie ihre Legendensymbole erzeugt.
Hüllen und gruppeninterne Linien sind neutral; die Punktfüllungen entsprechen
den vier Änderungstypen plus Start.
Reverts werden ausschließlich als exakte SHA1-Rücksprünge erkannt: Eine
Revision gilt als Revert, wenn sie einen früheren, SHA1-identischen
Artikelzustand wiederherstellt. Die zweidimensionale Projektion entscheidet
nicht über diese Einstufung.
Exakte Reverts erhalten auf der normalen Trajektorienkante keine Pfeile, damit
Hin- und Rückbewegungen nicht durch überlagerte Pfeilspitzen unleserlich werden.
Ein gepunkteter roter Rückbogen beginnt am Datenpunkt der rückgängig
gemachten Änderung und endet an der neuen Revert-Revision. Bei einer Folge
`Revision 1 → Revision 2 → zurück zu Revision 1` zeigt seine Pfeilspitze somit
von Position 2 zurück zu Position 1. Ein zusätzliches Rücksprung-Glyph mit
weißer Hintergrundmaske wird nicht mehr gezeichnet; der gepunktete Rückbogen
übernimmt die Kennzeichnung vollständig und der normale Revisionspunkt bleibt
sichtbar. Die SHA1-Gleichheit entscheidet unabhängig davon, ob die betroffenen
Revisionen von verschiedenen Autoren oder demselben Autor stammen; ein
Selbst-Revert wird daher ebenfalls rot dargestellt. Falls Projektion oder
gespeicherte Rundung einen exakten SHA1-Revert
geringfügig neben der wiederhergestellten Revision platzieren, wird nur dieser
exakte Revert-Punkt für die Anzeige auf die Koordinaten seiner
`restored_revision_id` gesetzt. Tritt dieselbe visuelle Rückverbindung in
gleicher oder entgegengesetzter Richtung mehrmals auf, werden eng benachbarte
Endpunkte innerhalb von fünf Prozent der Verbindungslänge als dasselbe Paar
behandelt. Ausschließlich der chronologisch letzte exakte Revert wird
gezeichnet. Der Bogen beginnt an der für dieses letzte Ereignis rückgängig
gemachten Revision und seine Pfeilspitze zeigt zur letzten Revert-Revision.
Frühere überlagerte Hin- oder Rückbögen derselben Verbindung werden
unterdrückt. Die Bögen
werden in Bildschirmkoordinaten berechnet und verändern die semantischen
MDS-Positionen nicht.

Reine Statusmeldungen ohne inhaltliche Änderung (`No content change`,
`Unchanged content`, `Content unchanged`) werden nicht als Kartenlabels
gerendert. Die zugehörigen Revisionen und Trajektorienkanten bleiben erhalten.

## Installation

Erforderlich sind Python 3, Git und für die Labelgenerierung ein OpenAI-API-Key.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Der API-Key liegt ausschließlich in der ignorierten Datei `.env`:

```text
OPENAI_API_KEY=sk-...
```

## Pipeline ausführen

### 1. Revisionsgeschichte laden

```bash
python src/wiki_api.py --article Palestine
```

Die XML-Datei wird unter `data/Palestine/Palestine.xml` gespeichert.

### 2. Projektion erzeugen

```bash
python src/main_pipeline.py \
  --article Palestine \
  --start 2012-12-02 \
  --end 2014-07-23 \
  --sample-strategy all \
  --output-name Palestine_small
```

Die Pipeline erzeugt eine label-freie Karte im selben interaktiven Grundgerüst
wie die finale Semantic-Zoom-Visualisierung, außerdem die Projektionsmetadaten,
Revisionsvektoren und die Kosinus-Distanzmatrix. Makrogruppen und ihre neutralen
Legendennamen werden deterministisch aus der Projektion gebildet; dafür werden
weder LLM- noch Label-Caches benötigt. Ausgaben werden fortlaufend nummeriert.

### 3. Finale Karte erstellen

Die finale Karte lässt sich aus den geprüften Hierarchie-, Kantenlabel- und
Revert-Artefakten ohne erneuten API-Aufruf reproduzieren:

```bash
python src/build_final.py
```

Der kurze Final-Build liest seine ausschließlich relativen Eingabepfade aus
`final_build.json`. Er benötigt nur die aktuelle Projektionsdatei und die
vollständig eingebettete, redaktionell geprüfte finale Hierarchie. Er verwendet
weder frühere HTML-Visualisierungen noch XML, Embeddings, Cache-Dateien oder
einen API-Zugang. Dadurch kann die eingecheckte `Palestine_final.html` auf einem
anderen Rechner bytegleich neu erzeugt werden.

Vollständige ISO-Zeitstempel aus den gespeicherten Kantenbeschreibungen werden
in sichtbaren Langlabels lokal auf `TT-MM-JJJJ` gekürzt. Die Quelldaten behalten
den vollständigen Zeitstempel; dafür ist kein erneuter API-Aufruf erforderlich.

Eine redaktionell geprüfte Labelpyramide kann im Hierarchie-JSON als fixiert
markiert werden. Ihre Makro-, Cluster- und Abschnittslabels werden dann auch bei
späteren Prompt- oder Logikänderungen unverändert übernommen. Die Fixierung
bleibt auch bei redaktionellen Korrekturen einzelner Kantenlabels erhalten. Sie
verfällt ausschließlich, wenn sich die Projektion und damit die geometrische
Grundlage der Hierarchie inhaltlich ändert.

## Bedienung

- Mausrad oder Trackpad: kontinuierlicher Zoom um den Zeiger von 1 % bis 200×;
  beim Herauszoomen wird die Geometrie zusammengeschoben, während Linien,
  Punkte und Schrift ihre Größe in Bildschirm-Pixeln behalten. Am äußersten
  Maßstab verdichtet sich die Karte zu einem Punkt mit dem priorisierten
  Artikel-Label aus dem jeweiligen Datensatz
- Ziehen: Karte verschieben
- Doppelklick: Gesamtansicht wiederherstellen
- Revisionspunkt: Revision-ID, Zeitpunkt und Autor anzeigen

Die HTML-Datei enthält Daten und Vega-Spezifikation, lädt Vega und Vega-Tooltip
aber über ein CDN und ist daher noch keine vollständig autarke Offline-Version.
Für eine Präsentation ohne Internet sollte zusätzlich ein statischer SVG- oder
PDF-Export bereitgehalten oder die beiden JavaScript-Bibliotheken lokal
eingebunden werden.

## Projektstruktur

```text
src/
├── build_final.py        reproduzierbarer Final-Build ohne API oder XML
├── wiki_api.py           Wikipedia-Revisionen laden
├── text_cleaning.py      Wikitext in Vergleichstext umwandeln
├── main_pipeline.py      Embeddings, MDS, HDBSCAN und Basiskarte
├── revert_detection.py   exakte SHA1-Reverts erkennen
├── revision_metadata.py  optionaler Metadaten-Fallback für ältere XML-Dateien
├── edge_labels.py        ein Terra-Label pro vollständigem Revisionspaar
├── apply_edge_labels.py  geprüfte Kantenlabels auf einen Bestand anwenden
├── apply_fixed_edge_label_overrides.py
│                          fixierte Kantenkorrekturen anwenden
├── apply_hierarchy_label_overrides.py
│                          geprüfte Hierarchielabels anwenden
├── refine_edge_labels_from_descriptions.py
│                          Kurzlabels aus Detailbeschreibungen nachbearbeiten
├── estimate_label_cost.py
│                          Umfang und Kosten eines Label-Laufs schätzen
├── semantic_zoom.py      Hierarchie und interaktive finale Karte
├── revision_keywords.py  optionale ausführliche Diff-Auditpipeline
├── git_word_diff.py      validierter Wortdiff für Auditdaten
├── llm_provider.py       OpenAI-Client und Modellkonfiguration
├── config.py             gemeinsame Pfade und CLI-Argumente
└── output_utils.py       nummerierte Ausgabedateien

tests/                    lokale, von Git ignorierte Tests
data/Palestine/           lokale, von Git ignorierte XML-Revisionsgeschichte
output/Palestine_final_data/
                          vier geprüfte Quelldateien des finalen Builds
output/Palestine_final.html
final_build.json          relative Eingaben des reproduzierbaren Final-Builds
```

## Lokale Tests

Der Ordner `tests/` ist bewusst über `.gitignore` ausgeschlossen und nicht Teil
des öffentlichen Repositorys. Ein frischer Klon enthält daher keine Tests. Sind
die lokalen Tests vorhanden, werden sie mit folgendem Befehl ausgeführt:

```bash
PYTHONPATH=src python -m pytest -q
```

Die lokalen Tests prüfen unter anderem Revisionsreihenfolge, Volltextkontext,
Diff-Rekonstruktion, Revert-Erkennung, feste Labelpositionen, Prioritäten und
Kollisionsfreiheit. Unabhängig davon enthält der eingecheckte finale
`v8`-Labelbestand 619 nichtleere Labels für 619 Kanten.
