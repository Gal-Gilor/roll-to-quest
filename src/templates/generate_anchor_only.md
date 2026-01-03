<task>
You are an expert in semantic query generation and D&D lore. Your task is to analyze a provided D&D-related text chunk and extract as many anchor queries as possible. These queries will be paired with the text chunk for training retrieval models.
</task>

<guidelines>
1.  **Understand the Goal:** Generate natural language questions or queries that this text chunk can answer. These will be used as "anchors" in anchor-positive training pairs where the provided text is the positive (relevant) document. **If the chunk is too short or lacks sufficient information, do not generate any anchors.**

2.  **Define Anchor:**
    *   **Anchor:** A question or query that a user might ask. Frame these as natural questions someone would actually search for - questions that this text chunk directly answers.
    *   Generate diverse query types: factual questions, "how-to" queries, definition requests, relationship queries, etc.
    *   Prioritize clarity and specificity - each anchor should have a clear answer in the provided text.
    *   Aim for 3-5 high-quality anchors per chunk (more if the chunk is rich in information).

3.  **Strategy for Anchor Generation:**
    *   **Read Thoroughly:** Analyze the D&D text chunk to identify all answerable questions.
    *   **Multiple Perspectives:** Generate queries from different angles - who, what, where, when, why, how.
    *   **Natural Language:** Write queries as real users would ask them, not as formal database queries.
    *   **Coverage:** Aim to cover the main topics and sub-topics in the text chunk.
    *   **Specificity:** Prefer specific questions over generic ones. "What weapon does Elara use?" is better than "What is in this text?"

4.  **Quality Guidelines:**
    *   Each anchor must be answerable from the provided text
    *   Anchors should be specific enough to distinguish this chunk from others
    *   Avoid overly generic queries like "What is D&D?" or "Tell me about this"
    *   Prefer questions that require the full context of the chunk
    *   Use natural, conversational language
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
</response_1>

<response_2>
Anchor: "What creature is terrorizing Oakhaven?"
</response_2>

<response_3>
Anchor: "Where did Elara find the shrine to Mielikki?"
</response_3>

<response_4>
Anchor: "What weapon does Elara carry?"
</response_4>

<response_5>
Anchor: "Who lives in the crumbling tower in the Whispering Woods?"
</response_5>
</responses>
</examples>
