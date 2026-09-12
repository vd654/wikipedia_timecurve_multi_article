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
- normalisierter Stress-1: 0,1462
- Shepard-Korrelation: 0,9750

`timecurve_002` bezeichnet den Projektionslauf. `terra_001` bezeichnet den
ersten Label-Lauf für genau diese Projektion. Die Nummern zählen unabhängige
Pipeline-Stufen und stehen nicht für vermischte Datensätze.

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

Die Chunks sind ausschließlich technische Eingabeeinheiten des
Embeddingmodells. Ein Punkt repräsentiert immer eine vollständige Revision;
Sätze, Abschnitte und Chunks werden nicht als eigene Punkte behandelt.

Die Distanz zweier Revisionen ist:

```text
1 - dot(L2-normalisierter Revisionsvektor A,
        L2-normalisierter Revisionsvektor B)
```

Es gibt keine zweite lexikalische Distanzmatrix, keine Hybridgewichtung, keine
Perzentil-Streckung und kein UMAP. Die Clusteranzahl wird nicht vorgegeben.

## Kantenlabels

Für jede der 619 Kanten erhält `gpt-5.6-terra` die beiden vollständigen
bereinigten Revisionstexte vor und nach der Änderung sowie Autor, Zeitstempel
und Bearbeitungskommentar beider Revisionen. Ein vorberechneter Diff wird nicht
übergeben; das Modell analysiert die beiden Volltexte direkt. Es erzeugt ein
unteilbares Label mit zwei bis drei Wörtern und eine ausführliche Beschreibung
der erkannten Änderung. Die Metadaten sind lediglich
nicht vertrauenswürdiger Zusatzkontext; der tatsächliche Textunterschied bleibt
maßgeblich. Wort-Diffs und Satzfenster werden nicht erzeugt oder gesendet.

Die Modellantworten werden nach jeder Kante gecacht. Ein abgebrochener Lauf
kann deshalb ohne erneute Aufrufe fortgesetzt werden. Beim späteren Zoomen
finden keine API-Aufrufe statt.

Die übergeordneten Ebenen werden vor dem Rendern mit derselben OpenAI-API aus
den gespeicherten Kantenlabels und Keywords ihres jeweiligen Bereichs erzeugt.
Terra formuliert für jeden zeitlichen Abschnitt, Cluster und jedes Makrothema
genau einen englischen Überbegriff. Die Antworten werden im Hierarchie-JSON
gecacht; mehrere Begriffe werden weder getrennt noch mit `·` kombiniert.

## Kartografische Hierarchie

```text
Artikel
└── Makrothema
    └── Themencluster
        └── zeitlicher Abschnitt
            └── Revisionskante
```

Die Hierarchie liefert lediglich Kandidaten und kartografische Prioritäten; sie
definiert keine Zoomstufen. Artikel-, Makro-, Cluster-, Abschnitts- und
Kantenlabels werden bei jedem Maßstab gemeinsam in Bildschirmkoordinaten
geprüft. Abstrakte Labels haben Vorrang, danach füllen detailliertere Labels
jeden ausreichend großen freien Bereich. Übergeordnete Labels besitzen eine
feste Ankerposition; jedes Kantenlabel wird in der Mitte des aktuell sichtbaren
Kantenabschnitts platziert. Alle Labels behalten eine konstante Schriftgröße.
Berühren sich zwei Textflächen, verschwindet das Label mit niedrigerer
Priorität. Es erscheint wieder, sobald Zoomen oder Verschieben genügend Platz
schafft. Ein Stabilitätsbonus verhindert Flackern bei kleinen Bewegungen.
Textgleiche Labelkandidaten werden bereits vor dem Rendering normalisiert und
dedupliziert. Dabei gewinnt zuerst die höhere Hierarchiepriorität; bei gleicher
Priorität entscheiden Importance, Revisionszahl und zuletzt die stabile Node-ID.
Jedes sichtbare Label besitzt eine vollständig deckende Textfüllung und eine
vollständig deckende weiße 4-px-Kontur. Ausgeblendete Kollisionskandidaten
werden ohne halbtransparente Zwischenphase entfernt.
Jede eigenständige Revisionskante bleibt dabei ein Labelkandidat; auch gleiche
Texte auf verschiedenen Kanten werden erst durch die räumliche Kollision und
nicht vorab durch eine globale Text-Deduplizierung entfernt.

Revisionspunkte verwenden stabile HDBSCAN-Farben; Noise bleibt grau. Punkte,
Abschnittsringe und Trajektorienlinien sind für die Detailansicht bewusst
kräftig dimensioniert. Auf jedem mindestens 34 Bildschirm-Pixel langen,
sichtbaren Teilstück einer normalen Kante zeigt ein Pfeil die chronologische
Revisionsrichtung. Wie bei Shaclens behält er beim Zoomen seine Bildschirmgröße
und sitzt bis zu 30 px hinter dem Anfang des aktuell sichtbaren Liniensegments.
Exakte Reverts erhalten keine Pfeile, damit Hin- und
Rückbewegungen nicht durch überlagerte Pfeilspitzen unleserlich werden. Am
Zielpunkt eines exakten Reverts ersetzt stattdessen ein aus einem zentrierten
Kreisbogen und einer Pfeilspitze aufgebautes rotes Rücksprung-Glyph den
normalen Revisionskreis; die zugehörige Kante endet dadurch direkt im Symbol
und behält die reguläre Trajektorienfarbe. An diesem Punkt wird auch ein möglicher
Abschnittsring unterdrückt. Da ein Revert häufig exakt auf der projizierten
Position einer älteren Revision landet, werden dort auch alle koordinatengleichen
normalen Kreise ausgeblendet und durch genau ein Glyph ersetzt. Ein kleines
Quadrat in Hintergrundfarbe liegt zusätzlich zwischen
Trajektoriengeometrie und Glyph; es maskiert mögliche Punkt- und Linienreste im
offenen Zentrum. Bestätigte Edit-War-Kanten erhalten einen
transparenten grünen Halo.

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

### 3. Revisionsmetadaten ergänzen

Das vorhandene XML bleibt unverändert. Autor, Zeitstempel und
Bearbeitungskommentar werden für die exakten Revisions-IDs der Projektion in
eine separate JSON-Datei geladen:

```bash
python src/revision_metadata.py \
  --projection \
    output/Palestine_small/Palestine_small_bert_timecurve_002_projection.json
```

### 4. Volltext-Kantenlabels erzeugen

```bash
python src/edge_labels.py \
  --article Palestine \
  --visualization \
    output/Palestine_small/Palestine_small_bert_timecurve_002_projection.json \
  --metadata \
    output/Palestine_small/Palestine_small_bert_timecurve_002_revision_metadata.json \
  --model gpt-5.6-terra
```

Das Modell erhält beide vollständigen bereinigten Revisionstexte sowie Autor,
Zeitstempel und Bearbeitungskommentar beider Revisionen. Es bekommt keinen
vorberechneten Diff, sondern vergleicht die Volltexte selbst. Metadaten gelten
dabei als nicht vertrauenswürdiger Zusatzkontext; der tatsächliche Textvergleich
bleibt maßgeblich. Der vorhandene Cache verhindert doppelte Modellaufrufe. Mit
`src/revision_keywords.py` steht zusätzlich eine ausführlichere Auditpipeline
mit Git-Wortdiff, Evidenz und Änderungsstatistiken zur Verfügung; sie ist für
die fertige Präsentationsdatei nicht erforderlich.

### 5. Finale Karte erstellen

```bash
python src/semantic_zoom.py \
  --projection \
    output/Palestine_small/Palestine_small_bert_timecurve_002_projection.json \
  --diff-labels \
    output/Palestine_small/Palestine_small_bert_timecurve_002_full_revision_edge_labels_gpt-5.6-terra_001.json \
  --article Palestine \
  --model gpt-5.6-terra \
  --skip-word-counts \
  --labels-output \
    output/Palestine_small/Palestine_small_bert_timecurve_002_semantic_zoom_full_revision_terra_001.json \
  --html-output output/Palestine_final.html
```

Der erste Lauf erzeugt die übergeordneten englischen Labels per API. Bei
unveränderten Eingaben werden sie aus dem Hierarchie-JSON wiederverwendet und
verursachen keine erneuten API-Kosten.

## Bedienung

- Mausrad oder Trackpad: kontinuierlicher Zoom um den Zeiger von 1 % bis 50×;
  beim Herauszoomen wird die Geometrie zusammengeschoben, während Linien,
  Punkte und Schrift ihre Größe in Bildschirm-Pixeln behalten. Am äußersten
  Maßstab verdichtet sich die Karte zu einem Punkt mit dem priorisierten
  Artikel-Label aus dem jeweiligen Datensatz
- Ziehen: Karte verschieben
- Doppelklick: Gesamtansicht wiederherstellen
- Revisionspunkt: Revision-ID, Zeitpunkt, Autor und Cluster anzeigen

Die HTML-Datei enthält Daten und Vega-Spezifikation, lädt Vega und Vega-Tooltip
aber über ein CDN und ist daher noch keine vollständig autarke Offline-Version.
Für eine Präsentation ohne Internet sollte zusätzlich ein statischer SVG- oder
PDF-Export bereitgehalten oder die beiden JavaScript-Bibliotheken lokal
eingebunden werden.

## Projektstruktur

```text
src/
├── wiki_api.py           Wikipedia-Revisionen laden
├── text_cleaning.py      Wikitext in Vergleichstext umwandeln
├── main_pipeline.py      Embeddings, MDS, HDBSCAN und Basiskarte
├── revert_detection.py   Reverts und Edit-War-Sitzungen erkennen
├── revision_metadata.py  Autor, Zeitstempel und Kommentare ergänzen
├── edge_labels.py        ein Terra-Label pro vollständigem Revisionspaar
├── semantic_zoom.py      Hierarchie und interaktive finale Karte
├── revision_keywords.py  optionale ausführliche Diff-Auditpipeline
├── git_word_diff.py      validierter Wortdiff für Auditdaten
├── llm_provider.py       OpenAI-Client und Modellkonfiguration
├── config.py             gemeinsame Pfade und CLI-Argumente
└── output_utils.py       nummerierte Ausgabedateien

tests/                    automatisierte Tests
data/Palestine/           lokale, von Git ignorierte XML-Revisionsgeschichte
output/Palestine_small/   Projektions-, Label- und Cache-Artefakte
output/Palestine_final.html
```

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

Die Tests prüfen unter anderem Revisionsreihenfolge, Volltextkontext,
Diff-Rekonstruktion, Revert-Erkennung, feste Labelpositionen, Prioritäten und
Kollisionsfreiheit. Die finale Artefaktprüfung bestätigt zusätzlich 619
nichtleere Labels für 619 Kanten.
