# Write hard search questions for D&D rules passages

You are writing training data for a search engine over the Dungeons & Dragons 5E rules (SRD 5.2.1). A keyword search already handles questions that reuse the words of a passage. You are writing the questions it fails on: ones a player would really ask, worded nothing like the passage.

## Input

A JSONL file. Each line is one passage:

- `passage_id`: copy it to your output unchanged.
- `label`: the kind of thing the passage is about (Spell, Monster, MagicItem, Feature, ...).
- `entity`: the name of that thing, such as "Fireball" or "Adult Red Dragon".
- `positive`: the passage. Its first line is the path of headers above it.

## What to write

For each passage, write two questions from someone who does not know or remember the entity's name and describes what they are looking for.

1. Do not use the entity's name or any word from it.
2. Do not reuse the passage's distinctive words. Say the same thing in everyday language:
   - "Paralyzed" becomes "frozen in place" or "unable to move".
   - "Necrotic damage" becomes "life-draining harm".
   - "saving throw" becomes "a roll to resist".
   - "Darkvision" becomes "seeing without light".
   - "Bonus Action" becomes "a quick extra move on my turn".
   Numbers may stay as they are.
3. Make it specific. Combine two or three facts from the passage that together point to this entry and no other. Many monsters share a trait and many spells share a damage type, so one fact is rarely enough. "Which undead claws people so they can't move, but can't do that to elves?" points to one monster. "Which monster can freeze you in place?" points to many.
4. The passage must contain the answer.
5. The question must make sense on its own. Never write "this spell", "the creature", "the passage" or similar.
6. Make the two questions different: build them on different facts, and start them differently. One can be a situation ("My character just got clawed by a corpse-eater and can't move. What happened?").
7. Write real questions ending with a question mark, 8 to 30 words long.

If a passage has too few facts to single it out without its name, write one question or none.

## Output

Write a JSONL file. One line per question, with exactly these keys:

```json
{"passage_id": "123-0", "question_type": "descriptive", "anchor": "Which undead claws people so they can't move, but can't do that to elves?"}
```

`question_type` is always "descriptive". No other text in the file. Process every passage in the input file.
