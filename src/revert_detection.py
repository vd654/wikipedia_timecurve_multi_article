"""Detect exact reverts and mutual-revert conflict in Wikipedia histories."""

from collections import Counter, defaultdict
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from rapidfuzz.distance import Levenshtein

from text_cleaning import affected_sentence_window, sentence_window_at_position

EDIT_WAR_MIN_EVENTS = 2
REVERT_ANALYSIS_VERSION = "2026-10-03-partial-revert-priority-v10"
PARTIAL_REVERT_MAX_DIRECTION_COSINE = -0.98
PARTIAL_REVERT_MIN_OUTGOING_COSINE = 0.98
PARTIAL_REVERT_MAX_OUTGOING_STEPS = 2
SEMANTIC_REVERT_MAX_DIRECTION_COSINE = -0.98
SEMANTIC_REVERT_MAX_RETURN_RATIO = 0.3
PARTIAL_REVERT_MAX_EVIDENCE_CHARS = 360
PARTIAL_REVERT_MAX_SENTENCE_CHARS = 2_000
EDIT_WAR_DEFINITION = (
    "Ein Edit War liegt bei mindestens zwei unmittelbar verketteten "
    "Gegenänderungsphasen zwischen mindestens zwei Autoren vor. Eine "
    "Ausgangsphase darf bis zu drei aufeinanderfolgende Trajektorienpunkte "
    "umfassen; die Gegenphase schließt unmittelbar an und führt sie ganz oder "
    "teilweise zurück. Ein einzelner "
    "Revert und Selbst-Reverts gelten nicht als Edit War. "
    "Ob parallel auf der "
    "Diskussionsseite kommuniziert wurde, ist ohne deren Versionsgeschichte "
    "nicht feststellbar."
)
PARTIAL_REVERT_DEFINITION = (
    "Ein partieller Revert beginnt mit einem Trajektorienschritt in eine "
    "Richtung. Eine oder mehrere unmittelbar folgende Revisionen führen die "
    "sichtbare zweidimensionale Trajektorie schrittweise in die Gegenrichtung "
    "zurück. Jeder Rückführungsschritt muss den Abstand zum Ausgangspunkt "
    "verkleinern; eine Mindestzahl übereinstimmender Inhaltswörter wird nicht "
    "verlangt."
)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element | None, name: str) -> str:
    if element is None:
        return ""
    for child in element:
        if local_name(child.tag) == name:
            return child.text or ""
    return ""


def probable_bot(user: str, tags: str = "") -> bool:
    """Return an audit-only bot hint for historical revisions."""
    folded_user = user.casefold()
    folded_tags = {tag.strip().casefold() for tag in tags.split("|") if tag.strip()}
    return "bot" in folded_tags or "bot" in folded_user


CONTENT_TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "which", "with",
}


def _content_tokens(words: list[str]) -> list[str]:
    """Normalize changed words while removing low-information glue words."""
    tokens = []
    for word in words:
        for token in CONTENT_TOKEN_RE.findall(word.casefold()):
            if token not in CONTENT_STOPWORDS and (len(token) > 2 or token.isdigit()):
                tokens.append(token)
    return tokens


def _snippet(
    words: list[str], max_chars: int = PARTIAL_REVERT_MAX_EVIDENCE_CHARS
) -> str:
    text = " ".join(words)
    text = re.sub(r"<!--.*?-->|<ref\b.*?</ref\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"[{}\[\]|'=]+", " ", text)
    text = re.sub(r"<[^>]+>|\s+", " ", text).strip()
    return text[:max_chars].strip()


def _affected_sentence(
    words: list[str], start: int, end: int
) -> str:
    """Return cleaned sentence context for a changed or zero-width span."""
    if start < end:
        window = affected_sentence_window(words, start, end)
    else:
        window = sentence_window_at_position(words, start)
    if window is None:
        return ""
    sentence_start, sentence_end = window
    return _snippet(
        words[sentence_start:sentence_end],
        max_chars=PARTIAL_REVERT_MAX_SENTENCE_CHARS,
    )


def build_change_signature(previous_text: str, current_text: str) -> dict:
    """Return compact changed passages for passage-level reversal matching."""
    previous_words = previous_text.split()
    current_words = current_text.split()
    units = []
    for opcode in Levenshtein.opcodes(previous_words, current_words):
        if opcode.tag == "equal":
            continue
        removed_words = previous_words[opcode.src_start:opcode.src_end]
        added_words = current_words[opcode.dest_start:opcode.dest_end]
        removed_tokens = _content_tokens(removed_words)
        added_tokens = _content_tokens(added_words)
        if not removed_tokens and not added_tokens:
            continue
        units.append({
            "added_tokens": added_tokens,
            "removed_tokens": removed_tokens,
            "added_text": _snippet(added_words),
            "removed_text": _snippet(removed_words),
            "previous_sentence": _affected_sentence(
                previous_words, opcode.src_start, opcode.src_end
            ),
            "current_sentence": _affected_sentence(
                current_words, opcode.dest_start, opcode.dest_end
            ),
        })
    return {"units": units}


def load_revert_history(
    xml_path: Path, *, include_text_changes: bool = False
) -> list[dict]:
    """Read revision metadata and optionally compact consecutive text diffs."""
    history = []
    previous_text = ""
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if local_name(element.tag) != "revision":
            continue
        contributor = next(
            (child for child in element if local_name(child.tag) == "contributor"),
            None,
        )
        user = (
            child_text(contributor, "username")
            or child_text(contributor, "ip")
            or "anonymous"
        )
        tags = child_text(element, "tags")
        revision = {
            "revision_id": child_text(element, "id") or "0",
            "timestamp": child_text(element, "timestamp"),
            "sha1": child_text(element, "sha1"),
            "user": user,
            "tags": tags,
            "probable_bot": probable_bot(user, tags),
            "history_index": len(history),
        }
        if include_text_changes:
            current_text = child_text(element, "text")
            revision["change_signature"] = (
                build_change_signature(previous_text, current_text)
                if history else {"units": []}
            )
            previous_text = current_text
        history.append(revision)
        element.clear()
    return history


def history_window_for_points(
    history: list[dict], points: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Slice full history to the plotted period and align sampled points."""
    by_revision_id = {
        str(revision["revision_id"]): index
        for index, revision in enumerate(history)
    }
    point_indices = []
    for point in points:
        revision_id = str(point["revision_id"])
        if revision_id not in by_revision_id:
            raise ValueError(
                f"Revisions-ID {revision_id} aus der Projektion fehlt im XML."
            )
        point_indices.append(by_revision_id[revision_id])
    if point_indices != sorted(point_indices):
        raise ValueError("Die Projektionspunkte sind nicht chronologisch geordnet.")
    start, end = point_indices[0], point_indices[-1]
    window = [revision.copy() for revision in history[start:end + 1]]
    for index, revision in enumerate(window):
        revision["history_index"] = index
    sampled = [
        {**point, "history_index": source_index - start}
        for point, source_index in zip(points, point_indices)
    ]
    return window, sampled


def editor_pair(left: str, right: str) -> tuple[str, str]:
    """Return a stable key for an unordered editor pair."""
    return tuple(sorted((left, right)))


def detect_sha1_reverts(revisions: list[dict]) -> list[dict]:
    """Find revisions that restore an earlier SHA1-identical article state."""
    latest_state = {}
    events = []

    for index, revision in enumerate(revisions):
        sha1 = revision.get("sha1", "")
        target_index = latest_state.get(sha1) if sha1 else None

        # A direct duplicate contains no intervening edit and is not a revert.
        if target_index is not None and target_index < index - 1:
            previous = revisions[index - 1]
            reverter = revision["user"]
            reverted_editor = previous["user"]
            event = {
                "revision_index": index,
                "revision_id": revision["revision_id"],
                "timestamp": revision["timestamp"],
                "reverter": reverter,
                "reverted_editor": reverted_editor,
                "reverted_revision_id": previous["revision_id"],
                "reverted_revision_index": index - 1,
                "directly_reverts_previous_revision": True,
                "restored_revision_id": revisions[target_index]["revision_id"],
                "restored_revision_index": target_index,
                "intervening_revision_count": index - target_index - 1,
                "pair": list(editor_pair(reverter, reverted_editor)),
                "revert_kind": "exact_sha1",
                "reversal_score": 1.0,
                "is_self_revert": reverter == reverted_editor,
                "involves_probable_bot": bool(
                    revision.get("probable_bot")
                    or previous.get("probable_bot")
                ),
            }
            signature = revision.get("change_signature", {})
            evidence = list(dict.fromkeys(
                text
                for unit in signature.get("units", [])
                for text in (unit.get("removed_text", ""), unit.get("added_text", ""))
                if text
            ))
            if evidence:
                event["conflict_passages"] = evidence
            sentences = list(dict.fromkeys(
                text
                for unit in signature.get("units", [])
                for text in (
                    unit.get("previous_sentence", ""),
                    unit.get("current_sentence", ""),
                )
                if text
            ))
            if sentences:
                event["conflict_sentences"] = sentences
            events.append(event)

        if sha1:
            latest_state[sha1] = index

    directions = defaultdict(set)
    for event in events:
        directions[tuple(event["pair"])].add(
            (event["reverter"], event["reverted_editor"])
        )
    mutual_pairs = {
        pair for pair, pair_directions in directions.items()
        if len(pair_directions) == 2
    }
    for event in events:
        event["mutual_in_full_history"] = tuple(event["pair"]) in mutual_pairs
    return events


def _signature_tokens(signature: dict, key: str) -> list[str]:
    return [
        token
        for unit in signature.get("units", [])
        for token in unit.get(key, [])
    ]


def _change_evidence(earlier_units: list[dict], later_units: list[dict]) -> tuple[list[str], list[str]]:
    evidence = list(dict.fromkeys(
        text
        for unit in earlier_units + later_units
        for text in (unit.get("removed_text", ""), unit.get("added_text", ""))
        if text
    ))
    sentences = list(dict.fromkeys(
        text
        for unit in earlier_units + later_units
        for text in (
            unit.get("previous_sentence", ""),
            unit.get("current_sentence", ""),
        )
        if text
    ))
    return evidence, sentences


def detect_partial_trajectory_reverts(
    revisions: list[dict],
    trajectory_vectors: dict[str, object] | None,
    *,
    max_direction_cosine: float = PARTIAL_REVERT_MAX_DIRECTION_COSINE,
    min_outgoing_cosine: float = PARTIAL_REVERT_MIN_OUTGOING_COSINE,
    max_outgoing_steps: int = PARTIAL_REVERT_MAX_OUTGOING_STEPS,
) -> list[dict]:
    """Detect consecutive piecewise returns in the visible 2D trajectory.

    An outgoing phase may contain one or two aligned steps (two or three
    points). Every immediately following return step must point almost opposite
    to the phase's aggregate vector and reduce its endpoint's distance to the
    phase origin. The longest qualifying outgoing phase wins for each return.
    """
    if not trajectory_vectors:
        return []

    candidates = []
    for start_index in range(len(revisions) - 2):
        try:
            start_vector = _vector(
                trajectory_vectors[str(revisions[start_index]["revision_id"])]
            )
            first_outgoing_vector = _vector(
                trajectory_vectors[
                    str(revisions[start_index + 1]["revision_id"])
                ]
            )
        except KeyError:
            continue

        first_outgoing = _vector_subtract(
            first_outgoing_vector, start_vector
        )
        if _vector_norm(first_outgoing) <= 1e-12:
            continue

        outgoing_end_indices = [start_index + 1]
        previous_outgoing_vector = first_outgoing_vector
        for outgoing_end_index in range(
            start_index + 2,
            min(len(revisions), start_index + max_outgoing_steps + 1),
        ):
            try:
                next_outgoing_vector = _vector(
                    trajectory_vectors[
                        str(revisions[outgoing_end_index]["revision_id"])
                    ]
                )
            except KeyError:
                break
            next_outgoing_step = _vector_subtract(
                next_outgoing_vector, previous_outgoing_vector
            )
            alignment = _direction_cosine(
                first_outgoing, next_outgoing_step
            )
            if alignment is None or alignment < min_outgoing_cosine:
                break
            outgoing_end_indices.append(outgoing_end_index)
            previous_outgoing_vector = next_outgoing_vector

        for outgoing_end_index in reversed(outgoing_end_indices):
            changed = revisions[outgoing_end_index]
            try:
                changed_vector = _vector(
                    trajectory_vectors[str(changed["revision_id"])]
                )
            except KeyError:
                continue
            outgoing = _vector_subtract(changed_vector, start_vector)
            outgoing_distance = _vector_norm(outgoing)
            return_indices = []
            direction_cosines = []
            previous_vector = changed_vector
            previous_distance = outgoing_distance
            current_index = outgoing_end_index + 1
            while current_index < len(revisions):
                current = revisions[current_index]
                try:
                    current_vector = _vector(
                        trajectory_vectors[str(current["revision_id"])]
                    )
                except KeyError:
                    break
                return_step = _vector_subtract(
                    current_vector, previous_vector
                )
                direction_cosine = _direction_cosine(
                    outgoing, return_step
                )
                current_distance = _vector_norm(
                    _vector_subtract(current_vector, start_vector)
                )
                if (
                    direction_cosine is None
                    or direction_cosine > max_direction_cosine
                    or current_distance >= previous_distance - 1e-12
                ):
                    break
                return_indices.append(current_index)
                direction_cosines.append(float(direction_cosine))
                previous_vector = current_vector
                previous_distance = current_distance
                current_index += 1

            if not return_indices:
                continue

            final_return = revisions[return_indices[-1]]
            outgoing_indices = list(
                range(start_index, outgoing_end_index + 1)
            )
            return_path_indices = [outgoing_end_index, *return_indices]
            full_path_indices = [*outgoing_indices, *return_indices]
            evidence, sentences = _change_evidence(
                [
                    unit
                    for index in range(start_index + 1, outgoing_end_index + 1)
                    for unit in revisions[index].get(
                        "change_signature", {}
                    ).get("units", [])
                ],
                [
                    unit
                    for index in return_indices
                    for unit in revisions[index].get(
                        "change_signature", {}
                    ).get("units", [])
                ],
            )
            return_ratio = previous_distance / outgoing_distance
            candidates.append({
                "revision_index": return_indices[-1],
                "revision_id": final_return["revision_id"],
                "timestamp": final_return["timestamp"],
                "reverter": final_return["user"],
                "reverted_editor": changed["user"],
                "reverted_revision_id": changed["revision_id"],
                "reverted_revision_index": outgoing_end_index,
                "directly_reverts_previous_revision": True,
                "restored_revision_id": revisions[start_index]["revision_id"],
                "restored_revision_index": start_index,
                "intervening_revision_count": 0,
                "pair": list(editor_pair(final_return["user"], changed["user"])),
                "revert_kind": "partial_trajectory",
                "reversal_score": float(max(0.0, 1.0 - return_ratio)),
                "trajectory_direction_cosine": max(direction_cosines),
                "trajectory_return_ratio": float(return_ratio),
                "trajectory_outgoing_path_revision_ids": [
                    revisions[index]["revision_id"]
                    for index in outgoing_indices
                ],
                "trajectory_path_revision_ids": [
                    revisions[index]["revision_id"]
                    for index in return_path_indices
                ],
                "trajectory_full_path_revision_ids": [
                    revisions[index]["revision_id"]
                    for index in full_path_indices
                ],
                "trajectory_outgoing_point_count": len(outgoing_indices),
                "trajectory_return_revision_count": len(return_indices),
                "trajectory_start_revision_index": start_index,
                "trajectory_first_return_revision_index": return_indices[0],
                "trajectory_final_revision_id": final_return["revision_id"],
                "trajectory_final_revision_index": return_indices[-1],
                "conflict_passages": evidence,
                "conflict_sentences": sentences,
                "involves_probable_bot": any(
                    revisions[index].get("probable_bot")
                    for index in full_path_indices
                ),
            })
            break

    best_by_return_start = {}
    for event in candidates:
        key = int(event["trajectory_first_return_revision_index"])
        rank = (
            int(event["trajectory_outgoing_point_count"]),
            int(event["trajectory_return_revision_count"]),
        )
        current = best_by_return_start.get(key)
        if current is None or rank > current[0]:
            best_by_return_start[key] = (rank, event)
    return sorted(
        (item[1] for item in best_by_return_start.values()),
        key=lambda event: (event["revision_index"], event["revision_id"]),
    )


def _vector(values) -> list[float]:
    return [float(value) for value in values]


def _vector_subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _vector_norm(values: list[float]) -> float:
    return sum(value * value for value in values) ** 0.5


def _direction_cosine(left: list[float], right: list[float]) -> float | None:
    denominator = _vector_norm(left) * _vector_norm(right)
    if denominator <= 1e-12:
        return None
    return float(sum(a * b for a, b in zip(left, right)) / denominator)


def _signature_topic_tokens(signature: dict) -> set[str]:
    return set(_signature_tokens(signature, "added_tokens")) | set(
        _signature_tokens(signature, "removed_tokens")
    )


def detect_semantic_counter_reverts(
    revisions: list[dict],
    semantic_vectors: dict[str, object] | None,
    *,
    max_direction_cosine: float = SEMANTIC_REVERT_MAX_DIRECTION_COSINE,
    max_return_ratio: float = SEMANTIC_REVERT_MAX_RETURN_RATIO,
) -> list[dict]:
    """Detect adjacent semantic A→B→A' counter-movements in full vectors.

    High-dimensional revision embeddings determine the counter-movement and
    return distance. Shared changed-passage tokens are retained as diagnostic
    information, but are not a classification threshold. The two-dimensional
    display projection is never used for classification.
    """
    if not semantic_vectors:
        return []
    events = []
    for index in range(2, len(revisions)):
        before, previous, current = revisions[index - 2:index + 1]
        if current["user"] == previous["user"]:
            continue
        try:
            before_vector = _vector(semantic_vectors[str(before["revision_id"])])
            previous_vector = _vector(semantic_vectors[str(previous["revision_id"])])
            current_vector = _vector(semantic_vectors[str(current["revision_id"])])
        except KeyError:
            continue
        previous_change = _vector_subtract(previous_vector, before_vector)
        counter_change = _vector_subtract(current_vector, previous_vector)
        direction_cosine = _direction_cosine(previous_change, counter_change)
        if direction_cosine is None or direction_cosine > max_direction_cosine:
            continue
        previous_step = _vector_norm(previous_change)
        current_step = _vector_norm(counter_change)
        return_distance = _vector_norm(
            _vector_subtract(current_vector, before_vector)
        )
        return_ratio = return_distance / max(previous_step, current_step, 1e-12)
        if return_ratio > max_return_ratio:
            continue
        previous_signature = previous.get("change_signature", {})
        current_signature = current.get("change_signature", {})
        shared_topic_tokens = (
            _signature_topic_tokens(previous_signature)
            & _signature_topic_tokens(current_signature)
        )
        evidence, sentences = _change_evidence(
            list(previous_signature.get("units", [])),
            list(current_signature.get("units", [])),
        )
        events.append({
            "revision_index": index,
            "revision_id": current["revision_id"],
            "timestamp": current["timestamp"],
            "reverter": current["user"],
            "reverted_editor": previous["user"],
            "reverted_revision_id": previous["revision_id"],
            "reverted_revision_index": index - 1,
            "directly_reverts_previous_revision": True,
            "restored_revision_id": None,
            "restored_revision_index": None,
            "intervening_revision_count": 0,
            "pair": list(editor_pair(current["user"], previous["user"])),
            "revert_kind": "semantic_text",
            "reversal_score": float(
                ((-direction_cosine) + max(0.0, 1.0 - return_ratio)) / 2.0
            ),
            "semantic_direction_cosine": float(direction_cosine),
            "semantic_return_ratio": float(return_ratio),
            "shared_topic_token_count": len(shared_topic_tokens),
            "conflict_passages": evidence,
            "conflict_sentences": sentences,
            "involves_probable_bot": bool(
                current.get("probable_bot") or previous.get("probable_bot")
            ),
        })
    return events


def merge_revert_events(
    exact_events: list[dict],
    partial_events: list[dict],
    semantic_events: list[dict] | None = None,
) -> list[dict]:
    """Prefer partial evidence over exact SHA1 evidence per revision."""
    partial = list(partial_events)
    partial_revision_ids = {
        str(event["revision_id"]) for event in partial
    }
    semantic = [
        event for event in (semantic_events or [])
        if str(event["revision_id"]) not in partial_revision_ids
    ]
    semantic_revision_ids = {
        str(event["revision_id"]) for event in semantic
    }
    exact = [
        event for event in exact_events
        if str(event["revision_id"]) not in partial_revision_ids
        and str(event["revision_id"]) not in semantic_revision_ids
    ]
    merged = partial + semantic + exact
    merged.sort(key=lambda event: (event["revision_index"], event["revision_id"]))
    directions = defaultdict(set)
    for event in merged:
        directions[tuple(event["pair"])].add(
            (event["reverter"], event["reverted_editor"])
        )
    for event in merged:
        event["mutual_in_full_history"] = (
            len(directions[tuple(event["pair"])]) == 2
        )
    return merged


def yasseri_m_measure(revisions: list[dict], events: list[dict]) -> dict:
    """Calculate M from all mutual reverts, excluding the strongest pair."""
    edit_counts = Counter(revision["user"] for revision in revisions)
    directions = defaultdict(set)
    for event in events:
        pair = tuple(event["pair"])
        directions[pair].add((event["reverter"], event["reverted_editor"]))

    mutual_pairs = {
        pair for pair, pair_directions in directions.items()
        if len(pair_directions) == 2
    }
    contributions = {
        pair: float(min(edit_counts[pair[0]], edit_counts[pair[1]]))
        for pair in mutual_pairs
    }
    mutual_events = [
        event for event in events
        if tuple(event["pair"]) in mutual_pairs
    ]

    involved_editors = {editor for pair in mutual_pairs for editor in pair}
    raw_sum = float(sum(contributions.values()))
    if contributions:
        top_pair, top_value = max(
            contributions.items(), key=lambda item: (item[1], item[0])
        )
    else:
        top_pair, top_value = None, 0.0

    censored_sum = raw_sum - float(top_value)
    score = float(len(involved_editors) * censored_sum)
    return {
        "value": score,
        "raw_mutual_revert_sum": raw_sum,
        "censored_mutual_revert_sum": censored_sum,
        "mutually_reverting_editor_count": len(involved_editors),
        "mutual_pair_count": len(mutual_pairs),
        "mutual_revert_count": len(mutual_events),
        "excluded_top_pair": list(top_pair) if top_pair else None,
        "excluded_top_pair_contribution": float(top_value),
        "mutual_pairs": [list(pair) for pair in sorted(mutual_pairs)],
    }


def gandica_c_measure(events: list[dict]) -> dict:
    """Calculate Gandica et al.'s C from all mutual-revert pairs."""
    directions = defaultdict(set)
    for event in events:
        pair = tuple(event["pair"])
        directions[pair].add((event["reverter"], event["reverted_editor"]))
    mutual_pairs = {
        pair for pair, pair_directions in directions.items()
        if len(pair_directions) == 2
    }
    counts = Counter(
        tuple(event["pair"])
        for event in events
        if tuple(event["pair"]) in mutual_pairs
    )
    involved_editors = {editor for pair in mutual_pairs for editor in pair}
    reversal_sum = int(sum(counts.values()))
    return {
        "value": float(len(involved_editors) * reversal_sum),
        "mutually_reverting_editor_count": len(involved_editors),
        "mutual_pair_count": len(mutual_pairs),
        "mutual_revert_count": reversal_sum,
        "mutual_pairs": [list(pair) for pair in sorted(mutual_pairs)],
        "pair_revert_counts": [
            {"pair": list(pair), "revert_count": counts[pair]}
            for pair in sorted(mutual_pairs)
        ],
    }


def detect_edit_war_sessions(
    events: list[dict],
    *,
    min_events: int = EDIT_WAR_MIN_EVENTS,
) -> list[dict]:
    """Group uninterrupted direct counter-edits into edit-war sessions.

    Every event or compound counter-phase must begin immediately after its
    outgoing phase. Eligible events form one session only while phase
    boundaries and the editor-to-editor revert chain are consecutive. At least
    two linked events and two different authors are required; self-reverts
    never count. A run may contain more than one editor pair. No time window or
    talk-page evidence is used at this stage.
    """
    eligible = []
    for event in events:
        direct_revert = event.get("directly_reverts_previous_revision")
        if direct_revert is None:
            reverted_index = event.get("reverted_revision_index")
            direct_revert = (
                reverted_index is not None
                and int(event["revision_index"]) - int(reverted_index) == 1
            )
        if not direct_revert:
            continue
        if event.get("reverter") == event.get("reverted_editor"):
            continue
        eligible.append(event)

    candidates = []
    ordered = sorted(
        eligible,
        key=lambda event: (event["revision_index"], event["revision_id"]),
    )
    runs = []
    current = []
    for event in ordered:
        if current:
            previous = current[-1]
            previous_end_index = int(previous.get(
                "trajectory_final_revision_index",
                previous["revision_index"],
            ))
            current_counter_start_index = int(event.get(
                "trajectory_first_return_revision_index",
                event["revision_index"],
            ))
            revision_gap = current_counter_start_index - previous_end_index
            chain_continues = (
                revision_gap == 1
                and event["reverted_editor"] == previous["reverter"]
            )
            if not chain_continues:
                runs.append(current)
                current = []
        current.append(event)
    if current:
        runs.append(current)

    for session_events in runs:
        participants = sorted({
            editor
            for event in session_events
            for editor in (event["reverter"], event["reverted_editor"])
        })
        if len(session_events) < min_events or len(participants) < 2:
            continue
        editor_pairs = sorted({
            tuple(event["pair"]) for event in session_events
        })
        conflict_passages = list(dict.fromkeys(
            passage
            for event in session_events
            for passage in event.get("conflict_passages", [])
            if passage
        ))
        conflict_sentences = list(dict.fromkeys(
            sentence
            for event in session_events
            for sentence in event.get("conflict_sentences", [])
            if sentence
        ))
        candidates.append({
            # ``pair`` remains as a compatibility/display field. For a
            # multi-author conflict it intentionally contains all participants.
            "pair": participants,
            "participants": participants,
            "participant_count": len(participants),
            "editor_pairs": [list(pair) for pair in editor_pairs],
            "events": session_events,
            "event_count": len(session_events),
            "start_revision_index": session_events[0].get(
                "trajectory_start_revision_index",
                session_events[0]["revision_index"],
            ),
            "end_revision_index": session_events[-1].get(
                "trajectory_final_revision_index",
                session_events[-1]["revision_index"],
            ),
            "start_timestamp": session_events[0]["timestamp"],
            "end_timestamp": session_events[-1]["timestamp"],
            "local_gandica_c": float(
                len(participants) * len(session_events)
            ),
            "revert_kinds": sorted({
                event.get("revert_kind", "exact_sha1")
                for event in session_events
            }),
            "mean_reversal_score": float(sum(
                event.get("reversal_score", 1.0) for event in session_events
            ) / len(session_events)),
            "conflict_passages": conflict_passages,
            "conflict_sentences": conflict_sentences,
        })

    candidates.sort(key=lambda session: (
        session["start_revision_index"],
        session["end_revision_index"],
        session["pair"],
    ))
    sessions = []
    for number, candidate in enumerate(candidates, start=1):
        session_id = f"edit-war-{number}"
        event_ids = []
        for event in candidate.pop("events"):
            event["edit_war_session_id"] = session_id
            event_ids.append(str(event["revision_id"]))
        sessions.append({
            "session_id": session_id,
            **candidate,
            "event_revision_ids": event_ids,
        })
    return sessions


def analyze_reverts(
    history: list[dict],
    sampled: list[dict],
    semantic_vectors: dict[str, object] | None = None,
) -> dict:
    """Return full-history revert evidence aligned to sampled curve points."""
    exact_events = detect_sha1_reverts(history)
    trajectory_vectors = {
        str(revision["revision_id"]): revision["mds"]
        for revision in sampled
        if revision.get("mds") is not None
    }
    partial_events = detect_partial_trajectory_reverts(
        history, trajectory_vectors
    )
    semantic_events = detect_semantic_counter_reverts(
        history, semantic_vectors
    )
    events = merge_revert_events(
        exact_events, partial_events, semantic_events
    )
    edit_war_sessions = detect_edit_war_sessions(events)
    sessions_by_id = {
        session["session_id"]: session for session in edit_war_sessions
    }
    point_statistics = []
    previous_cutoff = -1
    previous_c = 0.0

    for revision in sampled:
        cutoff = revision["history_index"]
        prefix_events = [
            event for event in events if event["revision_index"] <= cutoff
        ]
        interval_events = [
            event for event in prefix_events
            if previous_cutoff < event["revision_index"] <= cutoff
        ]
        interval_exact_events = [
            event for event in interval_events
            if event.get("revert_kind") == "exact_sha1"
        ]
        interval_partial_events = [
            event for event in interval_events
            if event.get("revert_kind") in {
                "partial_trajectory", "semantic_text"
            }
        ]
        measure = yasseri_m_measure(history[:cutoff + 1], prefix_events)
        conflict = gandica_c_measure(prefix_events)
        current_mutual_pairs = {
            tuple(pair) for pair in conflict["mutual_pairs"]
        }
        interval_mutual_events = [
            event for event in interval_events
            if tuple(event["pair"]) in current_mutual_pairs
        ]
        interval_pairs = sorted({
            tuple(event["pair"]) for event in interval_mutual_events
        })
        interval_edit_war_events = [
            event for event in interval_events
            if event.get("edit_war_session_id") in sessions_by_id
        ]
        interval_session_ids = sorted({
            event["edit_war_session_id"]
            for event in interval_edit_war_events
        })
        interval_sessions = [
            {
                "session_id": session_id,
                "pair": sessions_by_id[session_id]["pair"],
                "participants": sessions_by_id[session_id]["participants"],
                "participant_count": sessions_by_id[session_id][
                    "participant_count"
                ],
                "editor_pairs": sessions_by_id[session_id]["editor_pairs"],
                "event_count": sessions_by_id[session_id]["event_count"],
                "local_gandica_c": sessions_by_id[session_id]["local_gandica_c"],
                "revert_kinds": sessions_by_id[session_id]["revert_kinds"],
                "mean_reversal_score": sessions_by_id[session_id][
                    "mean_reversal_score"
                ],
                "conflict_passages": sessions_by_id[session_id][
                    "conflict_passages"
                ],
                "conflict_sentences": sessions_by_id[session_id].get(
                    "conflict_sentences", []
                ),
            }
            for session_id in interval_session_ids
        ]
        point_statistics.append({
            "interval_revert_count": len(interval_events),
            "interval_exact_sha1_revert_count": len(interval_exact_events),
            "interval_partial_text_revert_count": len(interval_partial_events),
            "interval_mutual_revert_count": len(interval_mutual_events),
            "interval_edit_war_pairs": [list(pair) for pair in interval_pairs],
            "interval_confirmed_edit_war_event_count": len(
                interval_edit_war_events
            ),
            "interval_confirmed_edit_war_sessions": interval_sessions,
            "cumulative_revert_count": len(prefix_events),
            "cumulative_yasseri_m": measure["value"],
            "cumulative_gandica_c": conflict["value"],
            "gandica_c_delta": conflict["value"] - previous_c,
        })
        previous_cutoff = cutoff
        previous_c = conflict["value"]

    final_measure = yasseri_m_measure(history, events)
    final_conflict = gandica_c_measure(events)
    return {
        "analysis_version": REVERT_ANALYSIS_VERSION,
        "method": (
            "exact SHA1, consecutive 2D trajectory returns, and "
            "high-dimensional semantic counter-movement conflict"
        ),
        "definition": (
            "A revision is a revert when it restores an earlier SHA1 state or "
            "directly returns from the preceding visible 2D trajectory step "
            "or semantically counters the preceding change."
        ),
        "partial_revert_definition": PARTIAL_REVERT_DEFINITION,
        "edit_war_definition": EDIT_WAR_DEFINITION,
        "total_revert_count": len(events),
        "exact_sha1_revert_count": sum(
            event.get("revert_kind") == "exact_sha1" for event in events
        ),
        "partial_text_revert_count": sum(
            event.get("revert_kind") in {
                "partial_trajectory", "semantic_text"
            }
            for event in events
        ),
        "partial_trajectory_revert_count": sum(
            event.get("revert_kind") == "partial_trajectory"
            for event in events
        ),
        "semantic_text_revert_count": sum(
            event.get("revert_kind") == "semantic_text" for event in events
        ),
        "total_mutual_revert_count": final_measure["mutual_revert_count"],
        "confirmed_edit_war_session_count": len(edit_war_sessions),
        "confirmed_edit_war_event_count": sum(
            session["event_count"] for session in edit_war_sessions
        ),
        "edit_war_session_parameters": {
            "minimum_events": EDIT_WAR_MIN_EVENTS,
            "minimum_authors": 2,
            "allows_multiple_editor_pairs": True,
            "requires_consecutive_revision_indexes": True,
            "requires_direct_revert_of_previous_revision": True,
            "requires_editor_chain_continuity": True,
            "proximity_rule": "strictly_consecutive_trajectory_phases",
            "talk_page_activity_checked": False,
            "partial_revert_detection_basis": "visible_2d_trajectory",
            "partial_revert_max_direction_cosine": (
                PARTIAL_REVERT_MAX_DIRECTION_COSINE
            ),
            "partial_revert_min_outgoing_cosine": (
                PARTIAL_REVERT_MIN_OUTGOING_COSINE
            ),
            "partial_revert_max_outgoing_points": (
                PARTIAL_REVERT_MAX_OUTGOING_STEPS + 1
            ),
            "partial_revert_min_matched_tokens": None,
            "partial_revert_requires_consecutive_revisions": True,
            "partial_revert_allows_piecewise_return": True,
            "partial_revert_requires_monotonic_return": True,
            "partial_revert_precedence_over_exact_sha1": True,
            "semantic_revert_max_direction_cosine": (
                SEMANTIC_REVERT_MAX_DIRECTION_COSINE
            ),
            "semantic_revert_max_return_ratio": (
                SEMANTIC_REVERT_MAX_RETURN_RATIO
            ),
            "semantic_revert_requires_shared_topic_tokens": False,
            "display_projection_used_for_partial_revert_detection": True,
            "display_projection_used_for_semantic_revert_detection": False,
            "exclude_probable_bots": False,
        },
        "probable_bot_revert_count": sum(
            event.get("involves_probable_bot", False) for event in events
        ),
        "excluded_probable_bot_revert_count": 0,
        "yasseri_m": final_measure,
        "gandica_c": final_conflict,
        "events": events,
        "edit_war_sessions": edit_war_sessions,
        "point_statistics": point_statistics,
    }
