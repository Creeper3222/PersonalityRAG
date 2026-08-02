from __future__ import annotations

import hashlib
import re
from typing import Any


_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_PUNCTUATION_RE = re.compile(
    r"^[\s,.;:!?'\"，。；：！？、（）()\[\]{}<>《》]+|"
    r"[\s,.;:!?'\"，。；：！？、（）()\[\]{}<>《》]+$"
)


def canonicalize(value: str) -> str:
    value = str(value or "")
    if not value:
        return ""
    normalized = _EDGE_PUNCTUATION_RE.sub("", value.strip())
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    if normalized.isascii():
        normalized = normalized.lower()
    return normalized


def dedupe(values: list[Any], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        key = canonicalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


class GraphBuilder:
    def __init__(
        self,
        max_topics: int = 6,
        max_participants: int = 8,
        max_facts: int = 8,
    ):
        self.max_topics = max_topics
        self.max_participants = max_participants
        self.max_facts = max_facts

    def build(
        self, memory_id: int, content: str, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        session_id = metadata.get("session_id")
        persona_id = metadata.get("persona_id")
        summary = str(metadata.get("canonical_summary") or content)
        participant_nodes = self._participant_nodes(metadata)
        participant_aliases = {
            canonicalize(value)
            for display_name, _canonical, extra in participant_nodes
            for value in [
                display_name,
                extra.get("sender_id"),
                *(extra.get("aliases") or []),
            ]
            if value
        }
        topics = [
            item
            for item in dedupe(
                list(metadata.get("topics") or []), self.max_topics
            )
            if not participant_nodes or canonicalize(item) not in participant_aliases
        ]
        facts = dedupe(list(metadata.get("key_facts") or []), self.max_facts)
        if not facts and summary:
            facts = [summary]

        nodes: dict[str, dict[str, Any]] = {}

        def add_node(
            node_type: str,
            value: str,
            extra=None,
            *,
            canonical_value: str | None = None,
        ) -> str:
            canonical = canonicalize(canonical_value or value)
            key = f"{node_type}:{canonical}"
            nodes[key] = {
                "node_key": key,
                "node_type": node_type,
                "value": value,
                "canonical_value": canonical,
                "metadata": extra or {},
            }
            return key

        topic_keys = [add_node("topic", item) for item in topics]
        participant_keys = [
            add_node(
                "person",
                display_name,
                extra,
                canonical_value=canonical_value,
            )
            for display_name, canonical_value, extra in participant_nodes
        ]
        fact_keys = [
            add_node("fact", item, {"summary": summary}) for item in facts
        ]
        entries: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        def entry(
            entry_type: str,
            text: str,
            node_keys: list[str],
            relation_type: str,
            confidence: float,
            edge_key: str | None = None,
        ):
            payload = (
                f"{entry_type}|{memory_id}|{relation_type}|"
                f"{'|'.join(node_keys)}|{text}"
            )
            entries.append(
                {
                    "entry_key": hashlib.sha1(
                        payload.encode("utf-8")
                    ).hexdigest(),
                    "source_memory_id": memory_id,
                    "session_id": session_id,
                    "persona_id": persona_id,
                    "entry_type": entry_type,
                    "relation_type": relation_type,
                    "content": text,
                    "node_keys": node_keys,
                    "edge_key": edge_key,
                    "metadata": {
                        "source_memory_id": memory_id,
                        "session_id": session_id,
                        "persona_id": persona_id,
                        "importance": metadata.get("importance", 0.5),
                        "create_time": metadata.get("create_time"),
                        "last_access_time": metadata.get("last_access_time"),
                        "canonical_summary": summary,
                        "summary_schema_version": metadata.get(
                            "summary_schema_version"
                        ),
                        "graph_confidence": confidence,
                        "source_window": metadata.get("source_window"),
                    },
                }
            )

        for key in fact_keys:
            value = nodes[key]["value"]
            entry("fact", f"Fact: {value}. Summary: {summary}", [key], "fact", 0.9)
        for key in topic_keys:
            value = nodes[key]["value"]
            entry(
                "topic",
                f"Topic: {value}. Summary: {summary}",
                [key],
                "topic",
                0.75,
            )
        for key in participant_keys:
            value = nodes[key]["value"]
            entry(
                "participant",
                f"Participant: {value}. Summary: {summary}",
                [key],
                "participant",
                0.7,
            )

        def add_edge(
            source: str,
            target: str,
            relation: str,
            confidence: float,
            text: str,
        ):
            edge_key = f"{source}|{relation}|{target}|{memory_id}"
            edges.append(
                {
                    "edge_key": edge_key,
                    "source_key": source,
                    "target_key": target,
                    "relation_type": relation,
                    "source_memory_id": memory_id,
                    "confidence": confidence,
                    "metadata": {"summary": summary},
                }
            )
            entry(
                "edge",
                text,
                [source, target],
                relation,
                confidence,
                edge_key,
            )

        for topic in topic_keys:
            for fact in fact_keys:
                add_edge(
                    topic,
                    fact,
                    "describes",
                    0.82,
                    f"Topic {nodes[topic]['value']} describes fact {nodes[fact]['value']}. Summary: {summary}",
                )
        for person in participant_keys:
            for fact in fact_keys:
                add_edge(
                    person,
                    fact,
                    "mentioned_in",
                    0.88,
                    f"Participant {nodes[person]['value']} is linked to fact {nodes[fact]['value']}. Summary: {summary}",
                )
        for index, first in enumerate(participant_keys):
            for second in participant_keys[index + 1 :]:
                add_edge(
                    first,
                    second,
                    "co_occurs_with",
                    0.7,
                    f"Participant {nodes[first]['value']} co-occurs with participant {nodes[second]['value']}. Summary: {summary}",
                )
        return {"nodes": list(nodes.values()), "edges": edges, "entries": entries}

    def _participant_nodes(
        self, metadata: dict[str, Any]
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Return LivingMemory 2.5.3 stable people with legacy fallback."""

        resolved: list[tuple[str, str, dict[str, Any]]] = []
        identities = metadata.get("participant_identities")
        if isinstance(identities, list):
            seen: set[str] = set()
            for raw in identities:
                if not isinstance(raw, dict):
                    continue
                sender_id = str(raw.get("sender_id") or "").strip()
                platform = (
                    str(raw.get("platform") or "unknown").strip().lower()
                    or "unknown"
                )
                identity_key = canonicalize(
                    str(raw.get("identity_key") or f"{platform}:{sender_id}")
                )
                display_name = str(
                    raw.get("display_name") or sender_id
                ).strip()
                if not identity_key or not display_name or identity_key in seen:
                    continue
                seen.add(identity_key)
                aliases = dedupe(
                    [
                        *list(raw.get("aliases") or []),
                        display_name,
                    ],
                    32,
                )
                resolved.append(
                    (
                        display_name,
                        f"account:{identity_key}",
                        {
                            "identity_key": identity_key,
                            "sender_id": sender_id,
                            "platform": platform,
                            "aliases": aliases,
                            "is_bot": bool(raw.get("is_bot", False)),
                        },
                    )
                )
                if len(resolved) >= self.max_participants:
                    break
        if resolved:
            return resolved

        return [
            (participant, canonicalize(participant), {})
            for participant in dedupe(
                list(metadata.get("participants") or []),
                self.max_participants,
            )
        ]
