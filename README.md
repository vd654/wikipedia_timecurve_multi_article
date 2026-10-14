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
gemeinsam in Bildschirmkoordinaten geprüft. Abstrakte Labels haben Vorrang,
danach füllen detailliertere Labels jeden ausreichend großen freien Bereich.
Makrothemen besitzen keine eigenen Flächen, Konturen oder Halo-Hintergründe;
sichtbar bleiben ihre Labels und ihre gemeinsame Farbe auf den zugehörigen
Clusterhüllen.
Übergeordnete Labels besitzen eine
feste Ankerposition; jedes Kantenlabel wird in der Mitte des aktuell sichtbaren
Kantenabschnitts platziert. Alle Labels behalten eine konstante Schriftgröße.
Jede sichtbare Kindebene soll mindestens einen neuen inhaltlichen Begriff oder
eine konkrete Änderungsrichtung hinzufügen. Kantenkurzlabels nennen deshalb
Gegenstand plus Handlung beziehungsweise Ergebnis; ihre ausführliche Variante
ergänzt Autor, Zeitpunkt und vollständigen Umfang. Eine Prüffunktion kann
inhaltlich unspezifische Kindlabels für die redaktionelle Kontrolle melden;
der normale Kartenaufbau bricht deswegen derzeit nicht automatisch ab.
Berühren sich zwei Textflächen, verschwindet das Label mit niedrigerer
Priorität. Es erscheint wieder, sobald Zoomen oder Verschieben genügend Platz
schafft. Ein Stabilitätsbonus verhindert Flackern bei kleinen Bewegungen.
Textgleiche übergeordnete Artikel-, Makro-, Cluster- und Abschnittslabels
werden vor dem Rendern zusammengeführt. Räumlich getrennte Revisionskanten
behalten dagegen ihre eigenen Beschriftungen, auch wenn deren Texte identisch
sind. Nur vollständig überlagerte Kanten erhalten an derselben Geometrie
insgesamt ein Label. Dabei gewinnt zuerst eine vorhandene ausführliche
Erklärung, danach die größere Wortänderung, der Informationsumfang, die
Labelkonfidenz und der semantische Sprung. Die übrigen Kanten bleiben als
Geometrie und Revisionsereignisse erhalten.
Jedes sichtbare Label besitzt eine vollständig deckende Textfüllung.
Übergeordnete Hierarchielabels verwenden eine weiße 4-px-Kontur,
Kantenbeschriftungen eine weiße 2-px-Kontur. Ausgeblendete
Kollisionskandidaten werden ohne halbtransparente Zwischenphase entfernt.
Jede eigenständige Revisionskante bleibt ebenfalls ein Labelkandidat. Es gibt
keine feste Zoomschwelle: Die Kollisionsprüfung versucht zuerst die
ausführliche mehrzeilige Beschreibung. Reicht der freie Bildschirmraum dafür
nicht aus, wird an derselben Position das kurze Kartenlabel versucht. Erst
wenn auch dieses nicht passt, wird die Kantenbeschriftung ausgeblendet.
Übergeordnete Labels werden zuerst platziert und verdrängen daher beide
Varianten. Die vollständigen Beschreibungen werden unabhängig von den kurzen
Kartenlabels eingebunden, sodass redaktionell gewählte Kurzlabels erhalten
bleiben. Im aktivierten Gruppenfokus gilt dieselbe Platzregel:
Es werden nur Kanten der ausgewählten Gruppe berücksichtigt, aber auch ihre
Beschriftungen müssen die allgemeine Kollisionsprüfung bestehen. Wo genügend
Platz vorhanden ist, erscheint die vollständige Beschreibung; andernfalls wird
das Kurzlabel versucht oder die Beschriftung bis zu einem stärkeren Zoom
ausgeblendet. Effektiv punktförmige Kanten verwenden ausschließlich das
Kurzlabel. Die ausführliche Beschreibung bleibt in den eingebetteten Daten
erhalten, wird aber nicht als eigener Tooltip angezeigt.

Revisionspunkte und Clusterhüllen verwenden höchstens sechs stabile
Makrogruppenfarben; zusammen mit dem neutralen Grau für Noise entstehen damit
insgesamt sieben Farbkategorien. Die feste Zuordnung der finalen `v8`-Karte
lautet: Palestinian Statehood `#D4A017`, Palestinian UN Status `#F8711A`,
Palestinian State Formation `#8B5A2B`, Palestinian Sovereignty `#B5179E`,
Palestinian State Development `#4A1942`, Palestinian State Designation
`#9B7BC8` und Noise `#A8A8A8`. Die konkreten Signalrot- und Signalgrüntöne
bleiben ausschließlich exakten beziehungsweise partiellen Reverts vorbehalten.
Noise-Revisionen bleiben wie alle anderen Revisionen kreisförmig und werden
neutral grau gezeichnet. Liegt ihre 2D-Position innerhalb der räumlichen
Gesamtfläche der ausgewählten Makrogruppe, bleibt der graue Punkt im
Gruppenfokus vollständig sichtbar. Zugehörige graue Trajektoriensegmente,
Richtungspfeile und platzierbare Kantenlabels bleiben ebenfalls aktiv; die
geometrische Überlagerung ändert die HDBSCAN-Zuordnung nicht.
Die konkrete
Clusteridentität wird weiterhin durch die direkten semantischen Clusterlabels
vermittelt. Die Farblegende führt deshalb nur Noise und die Makrogruppen auf.
Jede Clusterhülle umfasst sämtliche projizierten Revisionspunkte des jeweiligen
HDBSCAN-Clusters. Für reguläre zweidimensionale Punktmengen wird eine leicht
erweiterte konvexe Hülle gezeichnet. Fallen die Punkte in der Projektion auf
eine Linie oder praktisch dieselbe Position zusammen, verbindet stattdessen
ein dünner, anklickbarer Strich ihre unterscheidbaren Positionen. Zusätzliche
runde Endkappen werden nicht gezeichnet.
Punkte,
Abschnittsringe und Trajektorienlinien sind für die Detailansicht bewusst
kräftig dimensioniert. Auf jedem mindestens 34 Bildschirm-Pixel langen,
sichtbaren Teilstück einer normalen Kante zeigt ein Pfeil die chronologische
Revisionsrichtung. Wie bei Shaclens behält er beim Zoomen seine Bildschirmgröße
und sitzt bis zu 30 px hinter dem Anfang des aktuell sichtbaren Liniensegments.
Die festen Farben für Trajektorie, exakten Revert und partiellen Revert werden
aus denselben Farbkonstanten wie ihre Legendensymbole erzeugt. Punkt-, Hüllen- und
Abschnittsringfarben entsprechen der separat aufgeführten Makropalette.
Bot-Kennzeichnungen sind dabei nur Audit-Metadaten. Reverts von automatisierten
und menschlichen Konten werden für Richtungen, Sitzungen sowie Yasseri-M und
Gandica-C gleich behandelt; aus der Edit-War-Erkennung wird kein Ereignis wegen
eines Bot-Tags oder `bot` im Benutzernamen ausgeschlossen.
Eine Edit-War-Sitzung beginnt bereits mit einer direkten Gegenänderung zwischen
zwei Autoren: Die Änderung eines Autors führt die unmittelbar vorherige
Änderung des anderen ganz oder teilweise in die frühere inhaltliche Richtung
zurück. Beide Autoren müssen also nicht wechselseitig revertieren. Weitere
lückenlos folgende Gegenänderungen dürfen auch mehr als zwei Autoren und damit
mehrere Autorenpaare enthalten. Eine andere Revision dazwischen beendet die
Sitzung; Zeit- oder lockere Revisionsfenster werden nicht verwendet. Ob die
Beteiligten parallel auf der Diskussionsseite kommuniziert haben, lässt sich
ohne deren separate Versionsgeschichte nicht feststellen und fließt daher
nicht in die Erkennung ein.
Neben identischen SHA1-Rücksprüngen zählen auch direkt aufeinanderfolgende
partielle Gegenänderungen. Es gibt ausschließlich die Kategorien exakter und
partieller Revert. Wenn dieselbe Revision beide Regeln erfüllt, erhält der
exakte SHA1-Revert Vorrang und wird nur als exakter Revert gezählt und
dargestellt. Die zweidimensionale Trajektorienregel verlangt eine unmittelbare,
nahezu entgegengesetzte und zum Ausgangspunkt hin monotone Rückbewegung.
Partielle Reverts werden ausschließlich anhand dieser sichtbaren
zweidimensionalen Trajektorie bestimmt.

**Partieller Revert:** Löscht beziehungsweise revidiert nur die fehlerhaften
oder unerwünschten Teile einer Änderung; die korrekten und nicht betroffenen
Teile bleiben bestehen. Die Rückführung kann in mehreren unmittelbar
aufeinanderfolgenden Teilstücken erfolgen. Jeder Rückführungsschritt muss fast
entgegengesetzt zur Ausgangsbewegung verlaufen und den Abstand zum
Ausgangspunkt verkleinern. Eine Mindestzahl übereinstimmender Inhaltswörter
wird für diese Trajektorienregel nicht verlangt.
Exakte Reverts erhalten auf der normalen Trajektorienkante keine Pfeile, damit
Hin- und Rückbewegungen nicht durch überlagerte Pfeilspitzen unleserlich werden.
Ein gestrichelter roter Rückbogen beginnt am Datenpunkt der rückgängig
gemachten Änderung und endet an der neuen Revert-Revision. Bei einer Folge
`Revision 1 → Revision 2 → zurück zu Revision 1` zeigt seine Pfeilspitze somit
von Position 2 zurück zu Position 1. Ein zusätzliches Rücksprung-Glyph mit
weißer Hintergrundmaske wird nicht mehr gezeichnet; der gestrichelte Rückbogen
übernimmt die Kennzeichnung vollständig und der normale Revisionspunkt bleibt
sichtbar. Die SHA1-Gleichheit entscheidet unabhängig davon, ob die betroffenen
Revisionen von verschiedenen Autoren oder demselben Autor stammen; ein
Selbst-Revert wird daher ebenfalls rot dargestellt, zählt allein aber nicht als
Edit War. Falls Projektion oder gespeicherte Rundung einen exakten SHA1-Revert
geringfügig neben der wiederhergestellten Revision platzieren, wird nur dieser
exakte Revert-Punkt für die Anzeige auf die Koordinaten seiner
`restored_revision_id` gesetzt. Partielle Reverts werden nicht verschoben.
Partielle Textumkehrungen erhalten einen grünen, gestrichelten Rückbogen zum
neuen partiellen Zustand. Jeder Revert wird ausschließlich als exakter oder
partieller Revert dargestellt.
Treffen exakte und partielle Revert-Bögen auf dieselbe projizierte Verbindung,
werden sie auf gegenüberliegenden Seiten der Verbindung gezeichnet, damit
keine der beiden Kategorien die andere verdeckt.
Wiederholte Reverts derselben beiden projizierten Zustände werden paarweise mit
größerem Bogenradius gezeichnet. Auch innerhalb bestätigter Edit-War-Sitzungen
bleiben diese Bögen rot beziehungsweise grün. Die Bögen werden in
Bildschirmkoordinaten berechnet und verändern die semantischen MDS-Positionen
nicht.

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

Die Pipeline erzeugt die Basiskarte, die Projektionsmetadaten, die
Revisionsvektoren und die Kosinus-Distanzmatrix. Ausgaben werden fortlaufend
nummeriert.

### 3. Finale Karte erstellen

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
- Makrogruppe, zugehörigen Punkt, Clusterbereich oder Legendeneintrag anklicken:
  die Gruppe fokussieren, fremde Gruppen abdunkeln und ihre Clusterlabels
  ausblenden. Räumlich zugehörige Noise-Punkte bleiben vollständig sichtbar;
  auch ihre Trajektoriensegmente, Richtungspfeile und – bei genügend Platz –
  Kantenlabels werden eingeblendet. Erneuter Klick oder `Escape` setzt den
  Fokus zurück
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
├── revert_detection.py   Reverts und Edit-War-Sitzungen erkennen
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
output/Palestine_small/   Projektions- und Labelartefakte; Caches lokal ignoriert
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
