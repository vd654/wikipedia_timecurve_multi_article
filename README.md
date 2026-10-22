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
- 12 algorithmische Edit-War-Kandidaten, davon 9 semantisch bestätigt
- 23 als Edit War dargestellte Revisionskanten
- 9 technisch getrennte Edit-War-Sitzungen mit je genau einem Kartenlabel
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
→ algorithmische Edit-War-Kandidaten
→ zeitgleiche Talk-Evidenz und separate semantische Validierung
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
Makrothemen bleiben als semantische Labels und als interne Fokusgruppen
erhalten, besitzen aber keine eigene Farbcodierung und keinen Legendeneintrag.
Signalrot kennzeichnet exakte Reverts, Signalgrün partielle Reverts und
Signalblau bestätigte Edit-War-Sitzungen.
Noise-Revisionen bleiben wie alle anderen Revisionen kreisförmig und erhalten
dieselbe Änderungsfarbcodierung. Liegt ihre 2D-Position innerhalb der räumlichen
Gesamtfläche der ausgewählten Makrogruppe, bleibt der Punkt im
Gruppenfokus vollständig sichtbar. Zugehörige graue Trajektoriensegmente,
Richtungspfeile und platzierbare Kantenlabels bleiben ebenfalls aktiv; die
geometrische Überlagerung ändert die HDBSCAN-Zuordnung nicht.
Die konkrete Clusteridentität wird weiterhin durch die direkten semantischen
Clusterlabels vermittelt. Die Farblegende enthält keine Makrothemen mehr.
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
Die festen Farben für Trajektorie, Edit War, exakten Revert und partiellen
Revert werden aus denselben Farbkonstanten wie ihre Legendensymbole erzeugt.
Hüllen und gruppeninterne Linien sind neutral; die Punktfüllungen entsprechen
den vier Änderungstypen plus Start.
Bot-Kennzeichnungen sind dabei nur Audit-Metadaten. Reverts von automatisierten
und menschlichen Konten werden für Richtungen, Sitzungen sowie Yasseri-M und
Gandica-C gleich behandelt; aus der Edit-War-Erkennung wird kein Ereignis wegen
eines Bot-Tags oder `bot` im Benutzernamen ausgeschlossen.
Ein algorithmischer Edit-War-Kandidat beginnt bereits mit einer direkten Gegenänderung zwischen
zwei Autoren: Die Änderung eines Autors führt die unmittelbar vorherige
Änderung des anderen ganz oder teilweise in die frühere inhaltliche Richtung
zurück. Beide Autoren müssen also nicht wechselseitig revertieren. Weitere
lückenlos folgende Gegenänderungen dürfen auch mehr als zwei Autoren und damit
mehrere Autorenpaare enthalten. Eine andere Revision dazwischen beendet die
Sitzung; Zeit- oder lockere Revisionsfenster werden für die algorithmische
Erkennung nicht verwendet.

Die Kandidaten werden anschließend in einem getrennten LLM-Schritt semantisch
geprüft. Ein Kandidat wird nur angenommen, wenn mindestens zwei zugehörige
Artikelrevisionen dasselbe konkrete Thema betreffen, sich inhaltlich
entgegenstehen, eine substantielle Änderung bilden, durch die gespeicherten
Volltextvergleiche belegt sind und die Bewertung mindestens `0.75` erreicht.
Bearbeitungskommentare und `Talk:Palestine` sind ausschließlich unterstützende
Evidenz und können einen fehlenden Artikeltext-Beleg nicht ersetzen. Aus der
Diskussionsgeschichte werden pro Kandidat nur thematisch passende Revisionen
zwischen dessen erstem und letztem Zeitstempel verwendet. Eine fehlende
zeitgleiche Diskussion führt nicht zur Ablehnung. Die finale Palestine-Karte
bestätigt auf diese Weise 9 von 12 Kandidatensitzungen beziehungsweise 23 von
30 Kandidatenkanten. Die technisch erkannten Sitzungsgrenzen bleiben dabei
unverändert: Das LLM bestätigt oder verwirft ausschließlich die vorhandenen
Kandidaten und führt weder ähnliche Themen noch zeitlich getrennte Sitzungen
zusammen.

Für jede bestätigte Sitzung erzeugt derselbe separate Aufruf ein neutrales,
gegenstandsbezogenes Faktenlabel mit drei bis acht Wörtern und eine kurze
sachliche Erklärung. Account-Namen, Wertungen, Schuldzuweisungen und
Absichtszuschreibungen sind darin ausgeschlossen; wenn beide Positionen direkt
belegt sind, wird eine Form wie `A vs. B` bevorzugt. Revisionsgebundene Zitate,
Thementyp, Änderungsrichtung und Talk-Unterstützung dienen intern der Prüfung
und werden nicht als Schema auf der Karte ausgegeben. Das kurze Label ist für
die entfernte Kartenansicht bestimmt; beim Annähern wird wie bei anderen Kanten
eine belegorientierte Konflikterklärung verwendet. Sie nennt, jeweils durch
Semikolons getrennt, die kurze Zusammenfassung, die Anzahl der Gegenänderungen,
die beteiligten Revisions-IDs, den UTC-Zeitraum und zwei revisionsgebundene
Textstellen für die gegenüberstehenden Positionen. Ein gesamter Edit-War-
Sitzung erhält genau ein Kartenlabel, nicht je ein Label pro beteiligter
Kante. Inhaltlich ähnliche, aber technisch getrennte Sitzungen behalten jeweils
ihr eigenes Label und ihren eigenen Zeitraum. Beide Texte
erscheinen im selben Signalblau wie der zugehörige Edit-War-Bogen; bei
vollständig überlagerten Kanten und bei Kollisionen mit normalen
Kantenbeschreibungen hat das bestätigte Edit-War-Label Vorrang. Übergeordnete
Artikel-, Makro-, Cluster- und Abschnittslabels behalten ihre kartografische
Priorität.
Normale
Kantenlabels werden in diesem Validierungslauf ausschließlich aus dem
vorhandenen Bestand gelesen und nicht erneut an die API gesendet.
Neben identischen SHA1-Rücksprüngen zählen auch direkt aufeinanderfolgende
partielle Gegenänderungen. Technisch bleiben exakter und partieller Revert die
beiden Erkennungsmethoden. Wenn dieselbe Revision beide Regeln erfüllt, erhält
der exakte SHA1-Revert bei der Erkennung Vorrang. Gehört das Ereignis jedoch zu
einer bestätigten Edit-War-Sitzung, hat die Kategorie Edit War in der
Visualisierung Vorrang vor beiden Revert-Methoden. Die zweidimensionale
Trajektorienregel verlangt eine unmittelbare, nahezu entgegengesetzte und zum
Ausgangspunkt hin monotone Rückbewegung. Partielle Reverts werden
ausschließlich anhand dieser sichtbaren zweidimensionalen Trajektorie bestimmt.

**Partieller Revert:** Löscht beziehungsweise revidiert nur die fehlerhaften
oder unerwünschten Teile einer Änderung; die korrekten und nicht betroffenen
Teile bleiben bestehen. Die Rückführung kann in mehreren unmittelbar
aufeinanderfolgenden Teilstücken erfolgen. Jeder Rückführungsschritt muss fast
entgegengesetzt zur Ausgangsbewegung verlaufen und den Abstand zum
Ausgangspunkt verkleinern. Eine Mindestzahl übereinstimmender Inhaltswörter
wird für diese Trajektorienregel nicht verlangt.
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
Selbst-Revert wird daher ebenfalls rot dargestellt, zählt allein aber nicht als
Edit War. Falls Projektion oder gespeicherte Rundung einen exakten SHA1-Revert
geringfügig neben der wiederhergestellten Revision platzieren, wird nur dieser
exakte Revert-Punkt für die Anzeige auf die Koordinaten seiner
`restored_revision_id` gesetzt. Partielle Reverts werden nicht verschoben.
Partielle Textumkehrungen erhalten einen grünen, gestrichelten Rückbogen zum
neuen partiellen Zustand. Ereignisse einer bestätigten Edit-War-Sitzung werden
stattdessen vorrangig als blaue Strich-Punkt-Bögen (`-·-·-·-`) dargestellt;
ihre zugrunde liegende exakte oder partielle Erkennung bleibt in den Metadaten
und im Tooltip erhalten.
Treffen exakte und partielle Revert-Bögen auf dieselbe projizierte Verbindung,
werden sie auf gegenüberliegenden Seiten der Verbindung gezeichnet, damit
keine der beiden Kategorien die andere verdeckt.
Wiederholte Reverts derselben beiden projizierten Zustände werden paarweise mit
größerem Bogenradius gezeichnet. Edit-War-Bögen werden zuletzt und mit etwas
stärkerer Linienbreite gezeichnet, damit ihre Priorität sichtbar bleibt. Die
Bögen werden in Bildschirmkoordinaten berechnet und verändern die semantischen
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
python src/wiki_api.py \
  --article "Talk:Palestine" \
  --start 2003-06-06 \
  --end 2015-12-31 \
  --min-revision-chars 0
```

Die XML-Datei wird unter `data/Palestine/Palestine.xml` gespeichert.
Die separat ignorierte Diskussionsgeschichte liegt unter
`data/Talk%3APalestine/Talk%3APalestine.xml`.

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

Die separate Kandidatenvalidierung verwendet vorhandene Kantenlabels und ruft
das Modell ausschließlich für algorithmische Edit-War-Sitzungen auf:

```bash
python src/edit_war_validation.py \
  --article Palestine \
  --projection output/Palestine_small/Palestine_small_bert_timecurve_002_projection.json \
  --edge-labels output/Palestine_small/Palestine_small_bert_timecurve_002_full_revision_edge_labels_gpt-5.6-terra_v8.json \
  --talk-xml data/Talk%3APalestine/Talk%3APalestine.xml \
  --cache output/Palestine_small/Palestine_small_bert_timecurve_002_edit_war_talk_assessment_cache.json \
  --output output/Palestine_small/Palestine_small_bert_timecurve_002_edit_war_talk_validated_001.json

python src/apply_edit_war_labels.py \
  --projection output/Palestine_small/Palestine_small_bert_timecurve_002_projection.json \
  --hierarchy output/Palestine_small/Palestine_small_bert_timecurve_002_semantic_zoom_full_revision_terra_v8.json \
  --assessments output/Palestine_small/Palestine_small_bert_timecurve_002_edit_war_talk_validated_001.json \
  --html-output output/Palestine_final.html
```

Der Assessment-Cache ist von den normalen Kantenlabel-Caches getrennt.
Anschließend lässt sich die Karte wie bisher reproduzieren:

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
├── talk_evidence.py      zeitgleiche Talk-Revisionen auswählen
├── edit_war_validation.py
│                          nur algorithmische Kandidaten semantisch prüfen
├── apply_edit_war_labels.py
│                          bestätigte Konfliktlabels separat übernehmen
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
