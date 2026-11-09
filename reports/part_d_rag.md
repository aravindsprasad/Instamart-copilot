### D.1 Chunking (40 words, overlap 10 words)

| document | chunks | reference |
|---|---|---|
| refund_policy.txt | 4 | 4 |
| delivery_policy.txt | 3 | 3 |
| escalation_policy.txt | 3 | 3 |
| payment_policy.txt | 3 | 3 |
| privacy_policy.txt | 2 | 2 |

Total chunks: 15 (reference 15). Every chunk records its source file name. With overlap 0 there are 13 chunks.

### D.2 TF-IDF retrieval

Query `refund policy late delivery` -> top three: refund_policy.txt (0.272), refund_policy.txt (0.251), refund_policy.txt (0.245) (all three from refund_policy.txt, as required).

**TF and IDF.** Term frequency (TF) measures how often a word occurs inside one chunk, while inverse document frequency (IDF) measures how rare the word is across all chunks, so words that are frequent in a chunk but rare overall get the highest weights. A limitation is that TF-IDF matches exact words only: two phrasings with the same meaning but different words (for example "money back" versus "refund") get no similarity, and a question whose key words are split across a chunk boundary can retrieve the wrong chunk.

### D.3 Augment and answer

`answer(question)` retrieves three chunks, then sends two messages to `mock_llm.chat`: a system message containing the words POLICY ASSISTANT (answer only from the context, otherwise say the information could not be found) and a user message `CONTEXT:` + newline + the three chunks each prefixed with `[<source file>]` on its own line + newline `QUESTION:` + newline + the question. It returns the model text and the sources used.

Example: *How many days do I have to raise a refund request?* -> "A refund request must be raised within 7 days of the delivery date." (sources: refund_policy.txt, privacy_policy.txt, refund_policy.txt).

### D.4 Retrieval and grounding evaluation (14 questions)

| measure | overlap 10 | overlap 0 | reference (overlap 10) |
|---|---|---|---|
| answerable: expected_doc in top-3 sources | 12/12 | 12/12 | 12/12 |
| answerable: answer contains expected_fact | 12/12 | 10/12 | 12/12 |
| unanswerable: fallback 'I could not find...' | 2/2 | 2/2 | 2/2 |

With zero overlap 2 answerable question(s) break:

- *What delivery time makes a completed order eligible for a full refund?* (expected fact "more than 45 minutes" in refund_policy.txt): the answer was "receives a full refund of the order amount.". The sentence that holds the fact is cut by a chunk boundary: no single 40-word chunk contains the whole sentence ("If a completed order is delivered in more than 45 minutes, the customer receives a full refund of the order amount."), so the model only sees a fragment of it and answers with the wrong half.
- *How soon is money returned when a payment fails but is deducted?* (expected fact "48 hours" in payment_policy.txt): the answer was "If a payment fails but money is deducted, the amount is returned automatically 1000 must be approved by a human support lead before money is returned.". The sentence that holds the fact is cut by a chunk boundary: no single 40-word chunk contains the whole sentence ("If a payment fails but money is deducted, the amount is returned automatically within 48 hours."), so the model only sees a fragment of it and answers with the wrong half.

Overlap keeps the words around a chunk boundary in two neighbouring chunks, so an answer that straddles the boundary still appears whole in at least one chunk.
