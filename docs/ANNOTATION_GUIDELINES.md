# Manzil v2 — Annotation guidelines

## Status and scope
The 1,000-record v2 pilot is a **synthetic candidate dataset**. Its annotations were generated from construction metadata and programmatic span alignment. Structural checks do not establish semantic correctness. No record is certified as manually validated.

## Intent labels
- `NAVIGATE`: requests directions, a route, or travel to a destination.
- `SEARCH`: asks to find/locate information about a place, without necessarily requesting a route.
- `ETA_QUERY`: asks travel time or estimated arrival time.
- `OTHER`: location mention without a navigation/search/ETA request, or an unrelated statement.

Use pragmatic intent, not merely the presence of a place name.

## Entity labels
- `DEST`: explicit destination or named place.
- `REF`: landmark/origin/reference location.
- `REL`: spatial relation (near, opposite, behind, beside, between, etc.).
- `DESC`: descriptive phrase identifying an unnamed destination.
- `DIST`: distance expression.
- `DIR`: direction, turn or orientation.
- `TRANSIT`: transport mode.
- `CONTEXT`: anaphoric/contextual reference (that one, the place we discussed, wahi).

Annotate only labels expressed in the utterance. Do not infer an unspoken destination as `DEST`.

## BIO and offsets
Whitespace tokenization is used in the pilot. `B-X` starts an entity, `I-X` continues it, and `O` means outside any entity. Offsets are Unicode character offsets into the original query. Multiword spans should be contiguous. For production annotation, define a tokenizer consistently for punctuation, clitics and mixed scripts.

## Spatial graph
Represent every relation as a directed edge, for example:
```json
{"source":{"entity":"DEST","text":"..."}, "relation":"BEHIND", "target":{"entity":"REF","text":"..."}, "edge_type":"DEST_REL_REF"}
```
Nested relations require multiple edges. Do not invent edges unsupported by the utterance.

## Sentence-level semantics
Record destination, destination description, references, spatial edges, distance, direction, ambiguity, resolution status and whether clarification is required. Use null/empty values where the query does not provide the information.

## Ambiguity
- `CLEAR`: destination and constraints are sufficiently specified.
- `AMBIGUOUS`: multiple plausible destinations/relations or conflicting constraints.
- `UNDERSPECIFIED`: insufficient information to identify a unique destination.
- Resolution values: `EXPLICIT`, `DESCRIBED`, `CONTEXTUAL`, `UNRESOLVED`.

## Quality workflow
1. Two fluent annotators independently annotate intent, spans, graph edges and semantic resolution.
2. Compute agreement for intent, spans, relation edges, ambiguity and resolution.
3. Adjudicate disagreements; retain both raw annotations and adjudicated gold.
4. Freeze a test set before model selection. Annotators must not see model predictions during gold annotation.
5. Track provenance, annotator IDs, adjudication notes and guideline version.
