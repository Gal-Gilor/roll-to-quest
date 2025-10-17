<task>
You are an expert in semantic triplet generation and D&D lore. Your task is to analyze a provided D&D-related text chunk, and from it, extract as many (anchor, positive, negative) triplets as possible.
</task>

<guidelines>
1.  **Understand the Goal:** The primary objective is to identify semantic relationships within the text and categorize them into (Anchor, Positive, Negative) triplets. The text chunk provided represents one complete document in a retrieval corpus. If the chunk is too short or lacks sufficient information, do not generate any triplets.

2.  **Define Triplet Components:**
    *   **Anchor:** A question or query that a user might ask. Frame these as natural questions someone would actually search for - questions that this text chunk can answer.
    *   **Positive:** **The ENTIRE text chunk provided to you, word-for-word.**. DO NOT extract individual sentences or create fragments - include the complete chunk exactly as given.
    *   **Negative:** A passage that is semantically irrelevant or incorrect for the anchor. **Prioritize "hard negatives"**: passages that are deceptively similar to what the anchor seeks (e.g., same D&D domain, overlapping keywords, similar entities, or nearly correct information) but are actually semantically distinct or factually incorrect/irrelevant for that specific anchor. These negatives are most valuable for teaching fine-grained discrimination.

3.  **Strategy for Triplet Generation:**
    *   **First Pass - Anchor Identification:** Read through the D&D text and identify questions that this chunk can answer. Prioritize questions that have clear answers in the text and where you can craft good hard negatives.
    *   **Second Pass - Positive Assignment:** For each Anchor, the positive is ALWAYS the complete, entire chunk provided. Multiple different anchors will share the same positive - this is correct and expected!
    *   **Third Pass - Negative Identification:** For each Anchor, create hard negatives that are plausible but factually wrong. Make them challenging by using similar entities, locations, or contexts from the text, but with subtle factual errors that make them incorrect for that specific Anchor.

4.  **Verification:** The Positive (entire chunk) should answer the Anchor question. The Negative should seem plausible due to overlapping context but be factually incorrect or irrelevant for that specific Anchor. The most effective negatives will be those that are close enough to be tricky, but ultimately wrong.
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
Positive: "Elara, a valiant elven ranger, roamed the Whispering Woods, her longbow at the ready. She sought traces of the corrupted owlbear that had been terrorizing the nearby village of Oakhaven. Deep within the woods, she stumbled upon a forgotten shrine dedicated to the nature goddess, Mielikki. Nearby, a crumbling tower, lair of the goblin chieftain Grungnar, cast a long shadow."
Negative: "Lyria, a valiant elven ranger, roamed the Whispering Woods with her longbow at the ready."
</response_1>

<response_2>
Anchor: "What threatens Oakhaven?"
Positive: "Elara, a valiant elven ranger, roamed the Whispering Woods, her longbow at the ready. She sought traces of the corrupted owlbear that had been terrorizing the nearby village of Oakhaven. Deep within the woods, she stumbled upon a forgotten shrine dedicated to the nature goddess, Mielikki. Nearby, a crumbling tower, lair of the goblin chieftain Grungnar, cast a long shadow."
Negative: "The goblin chieftain Grungnar has been terrorizing the nearby village of Oakhaven from his crumbling tower lair."
</response_2>

<response_3>
Anchor: "Where did Elara find the shrine to Mielikki?"
Positive: "Elara, a valiant elven ranger, roamed the Whispering Woods, her longbow at the ready. She sought traces of the corrupted owlbear that had been terrorizing the nearby village of Oakhaven. Deep within the woods, she stumbled upon a forgotten shrine dedicated to the nature goddess, Mielikki. Nearby, a crumbling tower, lair of the goblin chieftain Grungnar, cast a long shadow."
Negative: "Near the village of Oakhaven, Elara stumbled upon a forgotten shrine dedicated to the nature goddess, Mielikki."
</response_3>
</responses>
</examples>