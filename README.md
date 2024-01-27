# miqu 70b

First model in the potential series.

## Prompt format: Mistral

```
<s> [INST] QUERY_1 [/INST] ANSWER_1</s> [INST] QUERY_2 [/INST] ANSWER_2</s>...
```

Beware that some backends (like llama.cpp) add bos already (by default), so you don't need prepend it yourself.

## Settings

DO NOT CHANGE ROPE SETTINGS. This model uses high freq base with 32k seen tokens, it should be fine for most tasks.

Only tested with temp 1 and top_p 0.95 with everything else disabled.