# Write search questions for D&D rules passages

You are writing training data for a search engine over the Dungeons & Dragons 5E rules (SRD 5.2.1). Players and Dungeon Masters type a question, and the engine must find the passage that answers it.

## Input

A JSONL file. Each line is one passage:

- `passage_id`: copy it to your output unchanged.
- `label`: the kind of thing the passage is about (Spell, Monster, MagicItem, Feature, Section, ...).
- `entity`: the name of that thing, such as "Fireball" or "Adult Red Dragon".
- `positive`: the passage. Its first line is the path of headers above it.

## What to write

For each passage, write three questions that the passage answers. Each has a different `question_type`:

1. `lookup`: a direct question about one specific fact in the passage. It names the entity.
   "How far away can I target a point with Fireball?"
2. `paraphrase`: a question about a different fact, in the words a player would use at the table, not the words of the passage. It names the entity. Replace the passage's terms with everyday ones: "saving throw" can become "resist", "Hit Points" can become "health", "Speed" can become "how fast".
   "If I upcast Fireball with a stronger slot, how much harder does it hit?"
3. `descriptive`: a question from someone who doesn't know or remember the entity's name and describes what they are looking for. It must NOT contain the entity's name. It must be specific enough that this passage is the clear answer.
   "What spell makes a huge explosion of flame that burns everyone in a wide area?"

## Rules

- Every question must make sense on its own. The reader does not see the passage. Never write "this spell", "the creature", "this feature", "the passage", "according to the text" or similar.
- For `label` "Section", the entity is a section title that players may not know. Name the topic in natural words instead ("carrying capacity", "underwater combat").
- The passage must contain the answer. Do not ask about anything it doesn't state.
- Ask about different facts in the three questions.
- Vary how questions start. Do not begin every question with "What".
- Write real questions ending with a question mark, 6 to 25 words long.
- If a passage states no fact worth asking about (for example it only introduces a list that is not included), write no questions for it.

## Output

Write a JSONL file. One line per question, with exactly these keys:

```json
{"passage_id": "123-0", "question_type": "lookup", "anchor": "How far away can I target a point with Fireball?"}
```

No other text in the file. Process every passage in the input file.
