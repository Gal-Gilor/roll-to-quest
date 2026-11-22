# Judge hard question-passage pairs for a D&D rules search engine

You are checking training data for a search engine over the Dungeons & Dragons 5E rules (SRD 5.2.1). Each pair is a question a player might type and the passage that should be found for it.

Every pair you see is one a keyword search got wrong: it ranked other passages above the intended one. That happens for two different reasons, and your job is to tell them apart.

- The question is good but worded unlike its passage. These are the most valuable training pairs. Accept them.
- The question is ambiguous: another passage answers it as well as the intended one, or better. Training on it would teach the engine that a correct passage is wrong. Reject them.

## Input

A JSONL file. Each line has:

- `row`: copy it to your output unchanged.
- `question_type`: `lookup` and `paraphrase` questions should name their subject. `descriptive` questions deliberately do not name it and describe it instead. `table` questions ask for a value in a table.
- `anchor`: the question.
- `positive`: the intended passage. Its first line is the path of headers above it.
- `competitor`: the passage of a different rules entry that the keyword search ranked highest.

## How to judge

Read the question first, as someone who cannot see any passage. Then read both passages.

Accept when all of these hold:

1. `positive` states the answer to the question.
2. `competitor` does not answer the question, or answers a different question that only shares words with it.
3. The question is specific enough that, among everything in the D&D rules you know, `positive` is the entry a rules expert would point to.

Reject with one of these reasons:

- `not_answered`: `positive` does not contain the answer.
- `competitor_answers`: `competitor` answers the question as well as `positive` or better.
- `too_generic`: the question fits many spells, monsters, items or rules, even if `competitor` is not one of them.
- `not_self_contained`: the question only makes sense with the passage in view.

Answer `unsure` when you cannot decide, for example when `positive` and `competitor` are near-identical entries (two dragons of different ages, the same feature in two classes) and the question does not say which one it means, or when you would need a rule that is in neither passage. Do not guess. An unsure verdict is reviewed by someone else.

## Output

Write a JSONL file with one line per input row and exactly these keys:

```json
{"row": 12, "verdict": "accept", "reason": ""}
{"row": 57, "verdict": "reject", "reason": "competitor_answers"}
{"row": 90, "verdict": "unsure", "reason": "Both passages are Epic Boon features that differ only by class; the question names no class."}
```

`reason` is empty for accepted pairs, one of the four codes for rejected pairs, and one sentence for unsure pairs. No other text in the file. Judge every row.
