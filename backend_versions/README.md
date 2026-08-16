# Backend-Versionen

## 2026-08-11 v014

`wikipedia_timecurve_semantic_zoom_2026-08-11_v014.zip` ergänzt zwischen
Artikelname und den 44 HDBSCAN-Clustern eine räumlich gebildete Makroebene mit
sieben offline erzeugten Oberbegriffen. Dadurch erscheinen die einzelnen
Clusterüberschriften erst beim weiteren Hineinzoomen. Alle Ebenen sind exklusiv;
die Vega-Labelplatzierung verhindert zusätzlich Kollisionen bei Überschriften
und Kantenkeywords. `Start` und `Ende` werden erst in den beiden feinsten Ebenen
an den tatsächlichen Endpunkten sichtbar. Der sichtbare Zoom-Hilfetext wurde
entfernt. Clusterüberschriften, Hüllen, Linien und Punkte verwenden nun exakt
dieselbe stabile HDBSCAN-Farbe.

Quellbaum-Hash:

```text
1dcd2c6a86b088f0615aaf471e28f3752dd915e543481a94d17560f534c2a797
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v014.sha256
```

## 2026-08-11 v013

`wikipedia_timecurve_semantic_zoom_2026-08-11_v013.zip` verbessert die lokale
Clusterzuordnung und berücksichtigt kleine Cluster. Innerhalb eines Clusters
liegende Trajektoriensegmente verwenden seine HDBSCAN-Farbe, während
clusterübergreifende Linien im Clusterzoom stark zurücktreten. Die transparente
Hülle wird etwas klarer gezeichnet. Statt einer festen Top-10-Grenze wächst das
Labelbudget mit der Zoomstufe; gleichzeitig sind Clusterüberschriften kompakter.

Quellbaum-Hash:

```text
5ec7ff255c5de36a34d9bef26e1ca6c4c68cc8ec18c49e0afcf72e55603de361
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v013.sha256
```

## 2026-08-11 v012

`wikipedia_timecurve_semantic_zoom_2026-08-11_v012.zip` macht die räumliche
Zuordnung der Überschriften sichtbar. Im Clusterzoom erhält jeder
Nicht-Noise-Cluster eine transparente konvexe Farbhülle; die Überschrift nutzt
einen dunklen Ton derselben stabilen HDBSCAN-Farbe. Trajektorie und Punkte
bleiben darüber sichtbar. Es werden weder Clusterzentren noch zusätzliche
Punkte erzeugt.

Quellbaum-Hash:

```text
0112d4e7e88bf81015e5c8d2d68e09568104edd477b95b49c73746f053b63e7f
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v012.sha256
```

## 2026-08-11 v011

`wikipedia_timecurve_semantic_zoom_2026-08-11_v011.zip` entfernt technische
Bezeichnungen wie `Cluster 1` und `Cluster 2` aus den sichtbaren
Clusterüberschriften. Angezeigt wird nur noch der semantische Oberbegriff. Die
HDBSCAN-IDs, Wahrscheinlichkeiten und stabilen Clusterfarben bleiben intern und
in den Tooltips vollständig erhalten.

Quellbaum-Hash:

```text
dcf1e09b4a0f6505e622e1119c8b3cec8f679d953c6d974e7966fa7cbaae1872
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v011.sha256
```

## 2026-08-11 v010

`wikipedia_timecurve_semantic_zoom_2026-08-11_v010.zip` stellt die ursprüngliche
HDBSCAN-Farbkodierung wieder her. Jeder Revisionspunkt erhält erneut die stabile
Golden-Angle-Farbe seiner Cluster-ID; Noise bleibt grau. Die Trajektorie selbst
bleibt neutral, damit ein Übergang zwischen zwei Clustern nicht fälschlich einem
einzigen Cluster zugeschrieben wird. Der Punkt-Tooltip zeigt außerdem den
HDBSCAN-Namen und die Clusterwahrscheinlichkeit.

Quellbaum-Hash:

```text
fc2b38e5a387db3b195faaaad013786e22caf78bde84fdc064ac4af346019426
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v010.sha256
```

## 2026-08-11 v009

`wikipedia_timecurve_semantic_zoom_2026-08-11_v009.zip` bewahrt die vorhandenen
HDBSCAN-Bezeichnungen `Cluster 1`, `Cluster 2` usw. Jeder Cluster erhält
zusätzlich seinen semantischen Oberbegriff. Beide stehen als zweizeiliges Label
am selben Clusteranker, sodass keine weitere Zoomstufe und kein zusätzlicher
Punkt entsteht.

Quellbaum-Hash:

```text
9e6a9a7f77ac54dfe85a7957e97506b3e4f7a585cfb91016645cdde6b3d35ea6
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v009.sha256
```

## 2026-08-11 v008

`wikipedia_timecurve_semantic_zoom_2026-08-11_v008.zip` ersetzt die
überlappenden Label-Fades durch vier exklusive Zoomintervalle. Artikelname,
Clusterüberschriften, Abschnittsüberschriften und Kantenkeywords können dadurch
nie gleichzeitig mit einer benachbarten Detailstufe erscheinen. Der kalibrierte
Wechsel von Cluster- zu Abschnittslabels liegt bei Zoomstufe `2.4`, der Wechsel
zu Kantenkeywords bei `4.15`.

Quellbaum-Hash:

```text
d73014669f78196605869846c69f26c7ac2908fbcd8c4525a528ea3af93e2ed6
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v008.sha256
```

## 2026-08-11 v007

`wikipedia_timecurve_semantic_zoom_2026-08-11_v007.zip` beschränkt jede
Trajektorienkante auf genau ein geerdetes Keyword. Kurze mehrteilige Begriffe
bleiben als ein Keyword erhalten, aber mehrere Kandidaten werden nicht länger
mit einem Trennzeichen zu einem Kantenlabel kombiniert.

Quellbaum-Hash:

```text
131995a6de9dbaba1b1daf5957f3fca755bf48a22d7416b4efc7833761094337
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v007.sha256
```

## 2026-08-11 v006

`wikipedia_timecurve_semantic_zoom_2026-08-11_v006.zip` reduziert die
Zoomansicht auf die semantisch notwendigen Zeichen. Der Artikelname und die
Clusterüberschriften besitzen keine künstlichen Mittelpunkt-Symbole mehr.
Cluster- und Abschnittslabels werden oberhalb ihrer Gruppe platziert und
verschwinden vollständig, bevor im Detailzoom die Änderungen exakt in der
Mitte ihrer Trajektorienkante erscheinen.

Die sichtbaren `+…/−… Wörter`-Angaben wurden aus Kantenlabels und Tooltips
entfernt; sie bleiben ausschließlich als Auditmetadaten in der gespeicherten
Hierarchie. Die Linienstärke codiert nun den semantischen Sprung.

Quellbaum-Hash:

```text
069281f693ef91766ebabb5164e9e7000cb94d251ebceca5b0554c01fde60bfc
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v006.sha256
```

## 2026-08-11 v005

`wikipedia_timecurve_semantic_zoom_2026-08-11_v005.zip` korrigiert die
semantische Zoominteraktion. Die äußerste Ansicht besitzt nun den Artikelnamen
„Palestine“ als echte Root-Ebene; danach folgen Cluster, chronologische
Abschnitte und konkrete Revisionskanten. Weiteres Herauszoomen als bis zur
Gesamtansicht ist begrenzt.

Für die Laufzeit wurden 619 verschachtelte Vega-Liniengruppen durch 619 direkte
Rule-Marks ersetzt. `autosize: fit` wurde entfernt, damit Labelwechsel nicht
mehr die Canvas-Größe und damit den Zoom neu berechnen. Während Wheel- und
Pan-Gesten pausiert die kollisionsbasierte Labelplatzierung und läuft erst nach
dem Ende der Interaktion einmal. Titel und Zoomstatus stehen in einer separaten
responsiven Kopfzeile und können nicht mehr mit Datenlabels kollidieren.

Quellbaum-Hash:

```text
d2598d488831c03609e5ceb21b8ad337edd83d82220d8b85fc7415cb266192d6
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_semantic_zoom_2026-08-11_v005.sha256
```

## 2026-08-11 v004

`wikipedia_timecurve_pipeline_keywords_semantic_zoom_2026-08-11_v004.zip`
ergänzt die gespeicherte Pipeline und Keyword-Extraktion um die einmalig offline
erzeugte semantische Labelpyramide. Enthalten sind Cluster-Oberbegriffe,
chronologische Abschnittslabels, konkrete Revisionskantenlabels, Git-basierte
`+…/−… Wörter`-Zahlen, parent-child-Beziehungen und eine native Vega-Timecurve
mit semantischem Zoom und kollisionsarmer Labelplatzierung. Die interaktive
Ansicht führt keine LLM-Aufrufe aus. Edit-War- und Vandalismusmuster sind in
diesem Stand ausdrücklich noch nicht enthalten.

Quellbaum-Hash:

```text
bdde512340aa8586bd6454cc363ab97ad1c6f5775a7fee2f190563f9999fc816
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_pipeline_keywords_semantic_zoom_2026-08-11_v004.sha256
```

## 2026-08-10 v003

`wikipedia_timecurve_pipeline_keywords_2026-08-10_v003.zip` erhöht das anhand
aller 619 vorhandenen Revisionspaare
kalibrierte Diff-Limit auf 100.000 Zeichen. Er ergänzt Positions- und
Opcode-Coverage, konditioniert den Truncation-Prompt, protokolliert validierte
Git-Warnungen ohne Fehlalarm und schreibt Snapshot-ID sowie Quellbaum-Hash in
jede neue Keyword-Datei.

Quellbaum-Hash:

```text
9e03f986671e48d1a206331a9c14ad08787224ba7d6b87640c37276e89f0f317
```

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_pipeline_keywords_2026-08-10_v003.sha256
```

## 2026-08-10 v002

`wikipedia_timecurve_pipeline_keywords_2026-08-10_v002.zip` erweitert die
Keyword-Extraktion um eine strikt validierte Wort-Diff-Engine auf Basis von
`git diff --no-index`. Enthalten sind außerdem die korrigierte
`difflib`-Baseline ohne `autojunk`, der Legacy-Modus, deduplizierte
Kontextgruppen, vollständige Diff-Metadaten und die zugehörigen Parser-,
Rekonstruktions- und Integrationstests.

Prüfung:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_pipeline_keywords_2026-08-10_v002.sha256
```

## 2026-08-10 v001

`wikipedia_timecurve_pipeline_keywords_2026-08-10_v001.zip` enthält den
verifizierten Stand der Wikipedia-Timecurve-Pipeline und der
Keyword-Extraktion vom 10. August 2026.

Enthalten sind:

- alle Python-Quellen unter `src/`,
- die Tests der Keyword-Extraktion,
- `README.md`, `requirements.txt` und `.gitignore`.

Nicht enthalten sind Zugangsdaten (`.env`), virtuelle Umgebungen, Rohdaten,
generierte Outputs und Python-Cachedateien.

Die Integritätsprüfung steht in der gleichnamigen `.sha256`-Datei. Vor dem
Entpacken kann sie aus dem Projektverzeichnis mit folgendem Befehl geprüft
werden:

```bash
shasum -a 256 -c backend_versions/wikipedia_timecurve_pipeline_keywords_2026-08-10_v001.sha256
```
