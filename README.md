# Wikipedia Timecurve

Das Projekt erzeugt eine semantische Timecurve aus Wikipedia-Revisionen und
kann anschließend Added-, Removed- und Focus-Stichwörter für die dargestellten
Revisionspaare generieren.

## Installation

```bash
pip install -r requirements.txt
```

Für die Stichwortgenerierung wird der OpenAI-Schlüssel in `.env` gespeichert:

```text
OPENAI_API_KEY=sk-...
```

## Dateien

```text
src/
├── wiki_api.py           Revisionen herunterladen
├── main_pipeline.py      semantische MDS-Timecurve erzeugen
├── revert_detection.py   exakte Reverts und Edit-War-Konflikte erkennen
├── revision_keywords.py  Added-, Removed- und Focus-Stichwörter
├── text_cleaning.py      MediaWiki-Text bereinigen
├── config.py             gemeinsame Kommandozeilenoptionen
└── output_utils.py       nummerierte Ausgabedateien
```

## 1. Revisionen herunterladen

```bash
python src/wiki_api.py --article "Palestine"
```

Die XML-Datei landet automatisch unter `data/Palestine/Palestine.xml`.

## 2. Timecurve erzeugen

```bash
python src/main_pipeline.py --article "Palestine"
```

Die Pipeline ist bewusst kurz:

```text
XML
→ alle Revisionen im gewählten Zeitraum
→ temporäre Token-Chunks
→ ein gemittelter, L2-normalisierter Vektor pro Revision
→ eine Kosinus-Distanzmatrix
→ klassische MDS in 2D
→ HDBSCAN direkt auf der 2D-Projektion
→ chronologische Kurve mit Clusterfarben
```

Standardmäßig werden alle Revisionen im gewählten Zeitraum verwendet. Eine
gröbere Sampling-Dichte muss ausdrücklich gewählt werden:

```bash
python src/main_pipeline.py --article "Palestine" --sample-strategy weekly
python src/main_pipeline.py --article "Palestine" --sample-strategy monthly
```

Ein begrenzter Lauf mit allen Revisionen kann getrennt gespeichert werden:

```bash
python src/main_pipeline.py --article "Palestine" \
  --start 2012-12-02 --end 2014-07-23 \
  --sample-strategy all --output-name "Palestine_small"
```

Dieser Lauf verwendet weiterhin `data/Palestine/Palestine.xml`, schreibt seine
Dateien aber separat nach `output/Palestine_small/`.

Die Artikel müssen wegen der Modellgrenze in maximal 256 Tokens lange Chunks
geteilt werden. Diese Chunks sind nur technische Encoding-Einheiten. Sie werden
nicht zwischen Revisionen zugeordnet und unmittelbar zu genau einem
Revisionsvektor gemittelt. Abschnitte und Sätze sind keine Analyseeinheiten.

Die Distanz zweier Revisionen ist direkt:

```text
1 - dot(L2-normalisierter Revisionsvektor A,
        L2-normalisierter Revisionsvektor B)
```

Es gibt keine lexikalische zweite Matrix, keine Hybridgewichtung, keine
Perzentil-Streckung und kein UMAP. HDBSCAN wird erst nach MDS direkt auf den
beiden sichtbaren Koordinaten ausgeführt. Eine Clusteranzahl wird nicht
vorgegeben; HDBSCAN bestimmt sie aus der Punktdichte und kann Noise markieren.

### Projektionsqualität

Die Projektionsdatei und die HTML-Ausgabe zeigen:

- normalisierten Stress-1 zwischen Kosinus- und 2D-Distanzen,
- Shepard-Korrelation zwischen Kosinus- und 2D-Distanzen.

Damit lässt sich prüfen, wie gut sichtbare Sprünge die ursprüngliche
Distanzmatrix wiedergeben.

### Ausgabedateien

Jeder Lauf wird fortlaufend nummeriert und erzeugt nur vier Dateien:

```text
output/Palestine/Palestine_bert_timecurve_003.html
output/Palestine/Palestine_bert_timecurve_003_projection.json
output/Palestine/Palestine_bert_timecurve_003_revision_embeddings.npy
output/Palestine/Palestine_bert_timecurve_003_cosine_distance_matrix.npy
```

Die HTML-Datei unterstützt Zoom mit Mausrad beziehungsweise Trackpad und Pan
durch Ziehen.

### Edit-War-Erkennung

Die Pipeline erkennt exakte Reverts daran, dass der SHA1-Hash einer Revision
einen früheren, nicht direkt vorhergehenden Artikelzustand wiederherstellt. Ein
einzelner Revert gilt noch nicht als Edit War: Erst wenn zwei menschliche
Editoren einander gegenseitig revertieren, wird das Intervall als Konflikt
klassifiziert. Diese Entscheidung folgt dem auf gegenseitigen Reverts
basierenden Konfliktparameter C nach Gandica, Sampaio dos Aidos und Carvalho
(`arXiv:1408.4362`). Vermutete Bot-Konten werden über XML-Tags und den
Kontonamen heuristisch ausgeschlossen.

Die Ergebnisse bleiben als Analysedaten in der Projektion erhalten. Die
semantische Zoom-Timecurve stellt Edit Wars jedoch nicht gesondert dar: Es gibt
keine rote Strichlinie, keinen Konfliktmarker und keine Edit-War-Legende.

## 3. Stichwörter erzeugen

```bash
python src/revision_keywords.py --article "Palestine"
```

Das Skript verwendet automatisch die neueste vollständige Visualisierung und
deren exakte Revisionsreihenfolge. Bei einer Weekly-Timecurve werden daher die
aufeinanderfolgenden Weekly-Punkte analysiert.

Standardmäßig wird der bereinigte Text mit `git diff --no-index` auf Wortebene
verglichen. Dafür schreibt das Skript jedes mit `str.split()` erzeugte Wort als
eigene Zeile in temporäre UTF-8-Dateien und verwendet einen kontextlosen
Histogram-Diff. Git muss daher als ausführbares Programm installiert sein; ein
Git-Repository ist nicht erforderlich.

Der Git-Parser behandelt fehlende Hunk-Anzahlen als `1` und rechnet
Null-Längen-Bereiche als Position vor einer Einfügung beziehungsweise nach
einer Löschung um. Entfernte und hinzugefügte Hunk-Zeilen werden gegen die
zugehörigen Wortbereiche geprüft. Anschließend muss eine Rekonstruktion aus
allen Opcodes exakt den Zieltext ergeben; andernfalls bricht die Analyse ab.

Änderungen mit überlappenden Kontextfenstern werden vor dem Rendern gruppiert.
Dabei erscheint der gemeinsame Kontext nur einmal, während getrennte
Änderungen ihre eigenen `[[ ... ]]`-Markierungen behalten.

Die Ausgabe enthält:

- `added_keywords`: 1 bis 8 Stichwörter für hinzugefügte Inhalte,
- `removed_keywords`: 1 bis 8 Stichwörter für entfernte Inhalte,
- `focus_keywords`: 5 Stichwörter zum Schwerpunkt der konkreten Änderung,
- `diff_truncated`: Kennzeichnung, ob der Modell-Diff gekürzt wurde,
- `diff_original_chars` und `diff_included_chars`: Zeichenumfang vor und nach
  der Begrenzung; der zweite Wert umfasst auch den Kürzungsmarker,
- `total_diff_opcodes`: Anzahl der vom Diff-Algorithmus erzeugten Änderungen,
- `included_diff_opcodes` und `omitted_diff_opcodes`: Anzahl der Opcodes in
  sichtbaren beziehungsweise vollständig ausgelassenen Units,
- `previous_word_count`, `current_word_count` sowie die Felder
  `first_included_*`, `last_included_*` und `last_included_*_fraction`:
  Wortposition und relative Reichweite der sichtbaren Änderungen,
- `total_change_units`, `included_change_units`, `omitted_change_units` und
  `shortened_change_units`: Umfang und Art der Kürzung.

Jedes Stichwort darf höchstens vier Wörter enthalten. `ADDED` und `REMOVED`
werden unabhängig behandelt. Wenn keine bereinigte Änderung vorhanden ist,
bleiben die Listen leer.

Modelldiffs sind standardmäßig auf 100.000 Zeichen begrenzt. Dieser Wert liegt
über dem am vorhandenen Korpus aus 619 Revisionspaaren gemessenen Maximum von
53.240 Diff-Zeichen. Passt dennoch nicht der
gesamte Diff, werden nur zusammenhängende Änderungs-Units vom Anfang
aufgenommen; nach der ersten nicht mehr passenden Unit wird abgebrochen. Ein
expliziter `[DIFF TRUNCATED: ...]`-Marker informiert das Modell über
Auslassungen. Der Truncation-Hinweis erscheint im Modellprompt ausschließlich
bei tatsächlich gekürzten Diffs. Ist bereits die erste Unit zu groß, wird sie an Wortgrenzen
gekürzt und zeigt jeweils Anfang und Ende der geänderten Passage. Jede Kürzung
wird außerdem mit den oben genannten Feldern im JSON protokolliert. Auf
Dateiebene enthält `truncated_revision_pair_count` die Gesamtzahl betroffener
Revisionspaare. Das Limit lässt sich mit `--max-diff-chars` ändern.

Die Diff-Methode ist pro Ausgabedatei mit `diff_engine`, `diff_algorithm`,
`diff_autojunk`, `git_version`, `diff_context_words`, `diff_input_unit` und
`diff_grouping` dokumentiert. `pipeline_snapshot_id` und `source_tree_sha256`
ordnen jede Keyword-Datei zusätzlich einem gesicherten Quellstand zu. Für
kontrollierte Vergleiche stehen drei Engines
zur Verfügung:

```bash
# Neuer Standard
python src/revision_keywords.py --article "Palestine" --diff-engine git

# Faire difflib-Baseline ohne automatische Junk-Heuristik
python src/revision_keywords.py --article "Palestine" --diff-engine difflib

# Reproduktion des früheren Verhaltens mit autojunk=True
python src/revision_keywords.py --article "Palestine" --diff-engine difflib-legacy
```

Der Git-Algorithmus und der Wortkontext lassen sich separat festlegen:

```bash
python src/revision_keywords.py --article "Palestine" \
  --git-diff-algorithm histogram \
  --diff-context-words 12
```

Bei einem Lauf wird immer nur die ausgewählte Engine an das Modell übergeben.

Das Standardmodell steht am Anfang von `src/revision_keywords.py`:

```python
MODEL = "gpt-5.6-terra"
```

Der System-Prompt lautet:

```text
Only use concepts explicitly supported by this revision.
Never use outside knowledge.
```

## 4. Semantische Zoom-Labels erzeugen

Nach Projektion und Keyword-Extraktion wird die Labelpyramide einmalig offline
gebaut. Beispiel für den vorhandenen `Palestine_small`-Datensatz:

```bash
python src/semantic_zoom.py \
  --projection output/Palestine_small/Palestine_small_bert_timecurve_002_projection.json \
  --keywords output/Palestine_small/Palestine_small_bert_timecurve_002_revision_keywords_gpt-5.6-terra_001.json \
  --article Palestine
```

Die Hierarchie hat fünf inhaltlich benannte Ebenen, zeigt aber keine technischen
Bezeichnungen wie „Level 1“:

- den Artikelnamen als einen einzigen Oberbegriff in der äußersten Ansicht,
- wenige räumliche Makrothemen als Brücke zur Clusterübersicht,
- Cluster-Oberbegriffe für die Übersicht,
- engere Begriffe für zusammenhängende chronologische Clusterabschnitte,
- konkrete Labels für den Unterschied zweier aufeinanderfolgender Revisionen.

Makrothemen werden auf der Makroebene fest im gewichteten Mittelpunkt ihrer
Punktgruppe platziert. Sie liegen damit direkt über dem beschriebenen Cluster
und werden nicht an einen einzelnen Revisionspunkt versetzt. Start-/End-,
Cluster- und Abschnittslabels behalten mehrere Ausweichpositionen und vermeiden
sowohl Revisionspunkte als auch bereits gesetzte Makrothemen.

Auf der Abschnittsebene umfasst eine transparente, clusterfarbene Hülle die
Revisionspunkte des jeweils beschriebenen zeitlichen Abschnitts. Nahezu
punkt- oder linienförmige Abschnitte erhalten stattdessen eine schmale
kapselartige Hülle entlang ihrer Trajektorie.

Übergangskanten, die keinem Abschnitt angehören, werden auf dieser Ebene
ausgeblendet. Dadurch enden gefüllte Kapseln sichtbar an ihren Abschnittspunkten
und wirken nicht wie verlängerte Linien. Auf der detaillierten Kantenebene sind
diese chronologischen Verbindungen wieder vollständig sichtbar.

HDBSCAN-Noise (`cluster_id = -1`) bleibt davon ausgenommen: Noise-Punkte werden
einheitlich grau dargestellt, aber nicht zu Abschnitten zusammengefasst und
erhalten weder Abschnittshüllen noch semantische Abschnittslabels. Ihre
einzelnen Übergänge werden erst auf der Kantenebene beschriftet.

Edit-Wars werden konservativ aus exakten SHA1-Reverts der vollständigen
Revisionsgeschichte bestimmt. Eine Sitzung umfasst dasselbe menschliche
Editorenpaar und bleibt zusammen, solange der nächste Revert innerhalb von 48
Stunden oder 15 Vollrevisionen folgt. Erst mindestens drei Ereignisse in beiden
Revert-Richtungen bestätigen eine Sitzung. Diese Analyse wird nicht als eigener
visueller Kanal in der Timecurve dargestellt.

Makro-, Cluster- und Abschnittsbegriffe werden in gebündelten LLM-Aufrufen ausschließlich
aus den bereits geerdeten Revisionskeywords erzeugt. Kantenlabels verwenden
jeweils genau ein konkretes Keyword des Revisionspaars und stehen exakt in der
Mitte der zugehörigen Trajektorienkante. Ein Keyword darf dabei ein kurzer
mehrteiliger Begriff wie `UN General Assembly` sein; mehrere Kandidaten werden
nie zu einem Kantenlabel kombiniert. Die Git-Wortzahlen bleiben als Auditmetadaten
im Hierarchie-JSON erhalten, werden aber weder als Label noch im Kanten-Tooltip
angezeigt. Bereits erzeugte LLM-Labels und Wortzahlen werden bei einem
unveränderten Datensatz wiederverwendet. Beim interaktiven Zoomen findet kein
LLM-Aufruf statt.

Die HTML-Ausgabe verwendet den Vega-Labelalgorithmus zur Kollisionsvermeidung
für Makro-, Cluster- und Abschnittsüberschriften. Diese stehen oberhalb ihrer Gruppe;
zusätzliche Root- oder Clusterpunkte werden nicht erzeugt. Fünf exklusive
Zoomwechsel bei `0.7`, `2.15`, `3.45` und `4.4` schalten zwischen
Artikelname, Makrothemen, Clusterüberschriften, Abschnittsüberschriften und den mittig auf
den Kanten platzierten Revisionslabels um. Dadurch sind Ober- und
Unterüberschriften nie gleichzeitig sichtbar. Freier Bildraum entscheidet
zusätzlich, welche Kandidaten innerhalb der aktiven Ebene platziert werden.
Dabei werden alle Kandidaten der aktiven Ebene berücksichtigt: Beim
Hineinzoomen vergrößern sich ihre Abstände im Bildschirmraum, sodass der
Labelalgorithmus automatisch mehr kollisionsfreie Begriffe zulässt. Innerhalb
einer Ebene werden wichtigere Begriffe zuerst platziert. Bereits sichtbare
Labels erhalten nach jedem Zoom- oder Pan-Vorgang einen Stabilitätsbonus, damit
kleine Ansichtsänderungen nicht zu unnötigem Flackern führen.
Alle Beschriftungen werden auf jeder Zoom-Ebene einheitlich fett dargestellt.
Auch die Kantenbegriffe werden nur am exakten Mittelpunkt angezeigt, wenn sie
dort kollisionsfrei platziert werden können. Verlaufen mehrere Revisionen auf
exakt derselben, richtungsunabhängigen Liniengeometrie, bleibt nur das Label des
Übergangs mit dem frühesten Zielrevision-Timestamp Kandidat; spätere Labels
entfallen, während alle Kanten und Tooltips vollständig erhalten bleiben.
Semantische Überschriften weichen
Revisionspunkten aus; Kantenlabels weichen sowohl den Punkten als auch den
semantischen Überschriften aus. Damit haben die übergeordneten Labels Vorrang,
falls zwei Labelarten künftig gleichzeitig eingeblendet werden. Die sichtbaren Überschriften bestehen ausschließlich aus den semantischen
Oberbegriffen; technische Bezeichnungen wie `Cluster 1` werden dort nicht
angezeigt. Cluster-ID und HDBSCAN-Zuordnung bleiben in den Daten und Tooltips
erhalten. `Start` und `Ende` erscheinen auf der zweiten Ebene gemeinsam mit den
Makrothemen an den tatsächlichen ersten und letzten Revisionspunkten. Die Revisionspunkte verwenden dieselbe stabile, aus der Cluster-ID
erzeugte Farbpalette wie die ursprüngliche Timecurve; Noise bleibt grau. Im
Clusterzoom umschließt zusätzlich eine schwach gefüllte, farbige konvexe Hülle
die zugehörigen Revisionspunkte. Überschriften verwenden exakt dieselbe
HDBSCAN-Farbe wie ihre Revisionspunkte. Dadurch bleibt die räumliche Zuordnung sichtbar, ohne
einen künstlichen Mittelpunkt zu erzeugen. Wenn Revisionspunkte verschiedener
Cluster exakt übereinanderliegen, werden auf der Abschnittsebene die tatsächlich
zum sichtbaren Abschnitt gehörenden Punkte zuletzt gezeichnet. So verdeckt kein
andersfarbiger, überlagerter Punkt die Farbzuordnung der Überschrift. Verbindungen innerhalb eines
Clusters erscheinen ebenfalls in dieser Farbe; clusterübergreifende
Trajektorien treten in dieser Zoomstufe zurück. Kleine Cluster werden bei
weiterem Zoom sichtbar, sobald ausreichend Platz vorhanden ist; eine
willkürliche feste Kandidatenobergrenze gibt es nicht. Die Linienstärke codiert den
semantischen Sprung; Kanten-Tooltips zeigen die beiden
Revisions-IDs, das Änderungslabel und den semantischen Sprung. Mausrad oder
Trackpad zoomen, Ziehen verschiebt die Ansicht, ein Doppelklick setzt sie zurück.
Weiteres Herauszoomen als bis zur Gesamtansicht wird verhindert. Während einer
laufenden Wheel-Geste pausiert nur die aufwendige Labelplatzierung und wird nach
dem Ende der Geste einmal neu berechnet. Die Trajektorie selbst bleibt dabei
sichtbar und reagiert unmittelbar. Die Canvas-Größe ändert sich ausschließlich
bei einer echten Fenstergrößenänderung, nicht bei jedem Zoomschritt.

Das Skript erzeugt neben der HTML-Datei ein JSON mit vollständiger Hierarchie,
Elternbeziehungen, Grounding-Begriffen, Git-Wortzahlen, Modell, Prompt,
Snapshot-ID und Quellhash. Mit `--no-llm` lässt sich eine rein deterministische
Vorschau bauen; `--skip-word-counts` überspringt nur die Git-Wortzählung.
