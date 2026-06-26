from __future__ import annotations

import hashlib
import re
from typing import Any


def canonicalize(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip().casefold()
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value)


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
        topics = dedupe(list(metadata.get("topics") or []), self.max_topics)
        participants = dedupe(
            list(metadata.get("participants") or []), self.max_participants
        )
        facts = dedupe(list(metadata.get("key_facts") or []), self.max_facts)
        if not facts and summary:
            facts = [summary]

        nodes: dict[str, dict[str, Any]] = {}

        def add_node(node_type: str, value: str, extra=None) -> str:
            canonical = canonicalize(value)
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
        participant_keys = [add_node("person", item) for item in participants]
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

