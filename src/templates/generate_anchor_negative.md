<task>
You are an expert in semantic triplet generation and D&D lore. Your task is to analyze a provided D&D-related text chunk, and from it, extract as many (anchor, negative) pairs as possible. These pairs will be used to train retrieval models by generating hard negatives for queries that the text chunk can answer.
</task>

<guidelines>
1.  **Understand the Goal:** The primary objective is to identify questions that the provided text chunk can answer (anchors), and create misleading but plausible incorrect passages (hard negatives) for each question. The text chunk provided represents one complete document in a retrieval corpus. If the chunk is too short or lacks sufficient information, do not generate any pairs.

2.  **Define Pair Components:**
    *   **Anchor:** A question or query that a user might ask. Frame these as natural questions someone would actually search for - questions that this text chunk can answer.
    *   **Negative:** A passage that is semantically irrelevant or incorrect for the anchor. **Prioritize "hard negatives"**: passages that are deceptively similar to what the anchor seeks (e.g., same D&D domain, overlapping keywords, similar entities, or nearly correct information) but are actually semantically distinct or factually incorrect/irrelevant for that specific anchor. These negatives are most valuable for teaching fine-grained discrimination.

3.  **Strategy for Pair Generation:**
    *   **First Pass - Anchor Identification:** Read through the D&D text and identify questions that this chunk can answer. Prioritize questions that have clear answers in the text and where you can craft good hard negatives.
    *   **Second Pass - Negative Identification:** For each Anchor, create hard negatives that are plausible but factually wrong. Make them challenging by using similar entities, locations, or contexts from the text, but with subtle factual errors that make them incorrect for that specific Anchor.

4.  **Verification:** The provided text chunk (not included in your output) will be used as the positive answer for each Anchor. The Negative should seem plausible due to overlapping context but be factually incorrect or irrelevant for that specific Anchor. The most effective negatives will be those that are close enough to be tricky, but ultimately wrong.
</guidelines>

<source_text>
{{ text }}
</source_text>

<examples>
<example_chunk>
Elara, a valiant elven ranger, roamed the Whispering Woods, her longbow at the ready. She sought traces of the corrupted owlbear that had been terrorizing the nearby village of Oakhaven. Deep within the woods, she stumbled upon a forgotten shrine dedicated to the nature goddess, Mielikki. Nearby, a crumbling tower, lair of the goblin chieftain Grungnar, cast a long shadow.
</example_chunk>

<responses>
<response_1>
Anchor: "Who is Elara?"
Negative: "Lyria, a valiant elven ranger, roamed the Whispering Woods with her longbow at the ready."
</response_1>

<response_2>
Anchor: "What threatens Oakhaven?"
Negative: "The goblin chieftain Grungnar has been terrorizing the nearby village of Oakhaven from his crumbling tower lair."
</response_2>

<response_3>
Anchor: "Where did Elara find the shrine to Mielikki?"
Negative: "Near the village of Oakhaven, Elara stumbled upon a forgotten shrine dedicated to the nature goddess, Mielikki."
</response_3>
</responses>
</examples>