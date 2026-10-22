# ProofOfRead

**Learn-to-Earn Reading Bounties with AI-Graded Comprehension and Answer-Copy Detection on GenLayer**

A sponsor funds a reward pool for anyone who reads an article and answers a
question about it correctly. An on-chain LLM grades each answer strictly
against the live article content, every correct reader is paid a fixed
reward until the slots run out, and a semantic check stops readers from
recycling answers that have already been submitted.

---

## What it does

1. A sponsor **posts a campaign**: a question, an article URL, a fixed
   reward per reader, a maximum number of reader slots, and a deadline. The
   sponsor funds exactly `reward_per_reader * max_readers` up front.
2. Any reader can **submit an answer** in their own words (at least 40
   characters). Each wallet gets **2 graded attempts** per campaign.
3. Before grading, the answer is embedded and compared against every answer
   already submitted for that campaign. A near-identical answer is rejected
   as a copy, and the rejection does **not** use up an attempt.
4. Otherwise the contract fetches the live article and asks an LLM whether
   the answer is correct, strictly from the article content. GenLayer's
   validator consensus (`gl.eq_principle.prompt_comparative`) ensures
   multiple independent LLM runs agree on the categorical
   `CORRECT`/`INCORRECT` verdict before it is accepted on-chain.
5. A `CORRECT` answer is **paid the fixed reward immediately**, once per
   wallet per campaign. When every slot is filled the campaign flips to
   `FULL`.
6. Safety exits: the sponsor can **close** the campaign early to recover the
   unspent budget, and anyone can trigger an **expired refund** to the
   sponsor once the deadline passes.

## Why this is different

The other bounty and escrow contracts in this series pay one winner
(TaskMatchEscrow, ReputationTaskMatch, FirstFixBounty) or split a pool
proportionally (QualityScoredBounty, PredictionPool). ProofOfRead is a
multi-winner, fixed-payout drip: every qualifying reader receives the same
reward until the budget is exhausted, which is the shape of real-world
"learn and earn" campaigns.

Because the answer text itself decides who gets paid, the contract treats it
as hostile input. The grading prompt labels the reader's answer as untrusted
data and instructs the model to ignore any instructions inside it, and the
model's output is strictly validated before any state change or transfer.

It also pairs the LLM verdict with a VecDB answer registry. Every graded
answer is embedded and stored, so a known answer (right or wrong) cannot be
pasted again, which blunts answer-sharing across wallets.

## Architecture

| Concern | Mechanism |
|---|---|
| Campaign state | Parallel `TreeMap[str, ...]` fields per campaign ID (sponsor, article URL, question, reward, slot count, rewarded count, budget, status, deadline) |
| Reader state | `TreeMap` keyed by `campaign_id:address` (lowercased) for attempts used and rewarded status |
| AI grading | `gl.nondet.web.render` fetches the article; `gl.nondet.exec_prompt` inside a closure, reconciled across validators via `gl.eq_principle.prompt_comparative` |
| Output safety | Grader output is validated (JSON parse, structure, field presence, type, and domain of the `verdict` field) before any state mutation or fund transfer |
| Prompt-injection hardening | The reader's answer is passed to the grader explicitly labeled as untrusted data to be graded, never instructions |
| Copy detection | Each graded answer is embedded (`all-MiniLM-L6-v2`, 384-dim) and stored in a `VecDB`, with a `registry_size` counter tracking how many records it holds. Before grading, the new answer is searched against **the entire registry** (`knn(embedding, registry_size)`) and the results are then filtered to the current campaign, so every prior answer for that campaign is always checked and answers from other campaigns can never crowd one out. A match under `COPY_DISTANCE_THRESHOLD = 0.05` within the same campaign triggers `REJECTED_COPY`, checked before any attempt is consumed |
| Payout | Fixed `reward_per_reader` per correct wallet, pure `u256` integer arithmetic, one reward per wallet per campaign |
| Fund safety | Funds only move on a `CORRECT` verdict (the fixed reward), `close_campaign` (sponsor, unspent budget only) or `claim_expired_refund` (deadline passed, unspent budget only). Unspent budget is always `budget - reward * readers_rewarded`, so paid-out funds can never be refunded a second time |

Funding is checked exactly (`gl.message.value == reward * max_readers`), so a
campaign can never be under-funded relative to its advertised slots.

Address-derived lookup keys are built with `str(gl.message.sender_address).lower()`
on the write side and `address.lower()` on the read side, because GenLayer's
`Address` string form is checksummed (mixed-case). No `Address`-typed method
parameters are used anywhere.

## Contract methods

### Write

- **`post_campaign(question: str, article_url: str, reward_per_reader_wei: int, max_readers: int, minutes_until_deadline: int) -> str`** *(payable)*
  Funds a campaign with exactly `reward_per_reader_wei * max_readers`.
  Returns the new `campaign_id` (e.g. `"R0"`).
- **`submit_answer(campaign_id: str, answer: str) -> str`**
  Copy-checks, then grades an answer. Returns `"CORRECT - you earned <x> wei. ..."`,
  `"INCORRECT - ..."`, or `"REJECTED_COPY - ..."`.
- **`close_campaign(campaign_id: str)`**
  Sponsor-only. Recovers the unspent budget and closes the campaign.
- **`claim_expired_refund(campaign_id: str)`**
  Anyone can trigger a refund of the unspent budget to the sponsor once the
  deadline has passed.

### View

- **`get_campaign(campaign_id: str) -> dict`**
  Full state of a campaign, including readers rewarded and unspent budget.
- **`get_my_status(campaign_id: str, address: str) -> dict`**
  A given address's attempts used, attempts left and rewarded flag.
- **`get_campaign_count() -> int`**
- **`get_answer_registry_size() -> int`**
  Total number of graded answers stored in the copy-detection registry.

## Tested end-to-end in GenLayer Studio

- `post_campaign` funded with 1 GEN (0.5 GEN x 2 slots) against a real
  python.org article, returning `"R0"`.
- **Wrong answer:** an answer claiming Python is compiled to machine code
  was graded **`INCORRECT`**, no payout, using attempt 1 of 2.
- **Copy detection:** resubmitting the identical wrong answer returned
  **`REJECTED_COPY`** without using up an attempt.
- **Correct answer:** an answer citing the article's readable, easy-to-learn
  syntax was graded **`CORRECT`** and paid
  `"500000000000000000 wei"` (0.5 GEN), `FINALIZED`/`SUCCESS`.
- `get_my_status` confirmed `attempts_used: "2"`, `attempts_left: "0"`,
  `rewarded: "yes"`.
- A further `submit_answer` from the rewarded wallet correctly **failed**
  with `"You were already rewarded for this campaign"`.
- `close_campaign` returned the unclaimed slot's 0.5 GEN to the sponsor, and
  `get_campaign` confirmed `status: "CLOSED"`, `readers_rewarded: "1"`,
  `unspent_budget: "500000000000000000"` (the amount that was unspent at
  the moment of closing).

### Fix round (steward-requested, Oct 2026)

A steward review flagged a gap in the copy check before accepting the
contract, fixed and retested:

- **Copy search is now exhaustive for the campaign.** The original check
  asked the vector store for only the 10 nearest records across *all*
  campaigns and filtered by campaign afterward. If 10 or more closer matches
  came from other campaigns, a prior answer for the current campaign could be
  pushed out of those 10, letting a copied answer through to grading and
  payment. Fixed by tracking a `registry_size` counter and searching the whole
  registry (`knn(embedding, registry_size)`), then filtering by campaign, so
  every relevant prior answer is always examined. The fixed `COPY_SEARCH_DEPTH`
  constant was removed, and a `get_answer_registry_size` view was added.

Verified in Studio on a fresh deployment with two campaigns:
- A wrong answer submitted to campaign `R0` was graded `INCORRECT`.
- The identical answer submitted to a second campaign, `R1`, was graded
  `INCORRECT` as well and was **not** rejected as a copy, confirming that
  answers from other campaigns do not count against it.
- Resubmitting the identical answer to `R0` was correctly rejected as
  **`REJECTED_COPY`**, even though `R1` now held an identical record, and
  without using up an attempt.
- A genuinely correct answer in `R0` was graded `CORRECT` and paid
  `"500000000000000000 wei"`.
- `get_answer_registry_size` returned `3`: the `R0` wrong answer, the `R1`
  wrong answer and the `R0` correct answer. The rejected copy is not stored.
- All test transactions reached `FINALIZED`/`SUCCESS` with validator
  agreement.

## Deployment

```
# v0.2.16
# { "Seq": [
#     { "Depends": "py-lib-genlayer-embeddings:09h0i209wrzh4xzq86f79c60x0ifs7xcjwl53ysrnw06i54ddxyi" },
#     { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
# ] }
```

Requires the `genlayer-embeddings` dependency for `VecDB` and
`SentenceTransformer`. Constructor takes no arguments. Deploy directly in
GenLayer Studio or via `genlayer-js` against Studionet / Bradbury testnet.

## Notes on GenVM quirks discovered while building this

- Looking up a possibly-missing `TreeMap` key is done with a small
  `try`/`except` helper rather than relying on a dict-style `.get()`, which
  keeps behavior explicit for keys that have never been written.
- The copy check runs entirely on deterministic inputs (the reader's own
  text, embedded locally), so it needs no equivalence principle. Only the
  article-grading step is non-deterministic and goes through consensus.
- Returning a string from a write method (instead of raising) is how a
  non-fatal outcome such as `REJECTED_COPY` or `INCORRECT` is reported
  without reverting. Raising or asserting reverts the whole transaction,
  which is what keeps the attempt counter and registry consistent when the
  grader output is malformed.

## Notes on design decisions

- The grader's output is validated (JSON structure, field presence, type,
  and domain of the `verdict` field) before any state change or fund
  transfer, the same safety pattern applied across every contract in this
  series after it was flagged as a required fix on an earlier contract
  (SemanticDisputeRouter).
- Copy detection is deliberately tight (distance under 0.05): independent
  readers who give correct answers in different words will both pass. It
  blocks near-verbatim recycling of a known answer, not sybil wallets in
  general.
- Copy detection is scoped per campaign by searching the full registry and
  filtering by campaign, rather than by taking a fixed number of nearest
  records. A fixed cutoff would be unsafe, because enough close matches from
  other campaigns could push a prior answer for the current campaign out of
  the results and let a copy through to grading and payment. The trade-off is
  that each submission's search cost grows with the total number of graded
  answers stored across all campaigns, which is acceptable at this contract's
  scale. A per-campaign index would be the next step for a much larger
  deployment.
- A rejected copy costs the reader nothing, since it consumes no attempt,
  while every graded answer (correct or not) is added to the registry.

## License

MIT
