# Claim extraction schema

Input: a file under data/extract_input/. Each block is one atomic unit: a transcript turn or an
email/report message. The bracketed id at the start of the block is the unit_id. Use it verbatim.

Output: JSON Lines, one claim per line, to data/claims_raw/<same basename>.jsonl. No prose, no
markdown fences, nothing but JSON lines. Fields:

- claim_id: "<unit_id>#<n>" with n starting at 1 within that unit
- unit_id: the unit the claim comes from. A claim must come from ONE unit. If a statement is spread
  over consecutive turns by the same speaker, attach it to the turn where the key value or verb is.
- kind: one of fact | proposal | agreement | rejection | decision | action | correction | withdrawal | recollection | question
    fact         = stated as the case ("shelf life is populated on forty-eight percent")
    proposal     = someone suggests or asks for something ("I suggest we drop the field")
    agreement    = someone accepts a specific proposal ("Confirmed, OP_ID is dropped from the next run")
    rejection    = someone declines a specific proposal
    decision     = a decision recorded as made (by whom)
    action       = a commitment to do something (by whom, by when if stated)
    correction   = the speaker corrects an earlier statement (theirs or another's) and gives the right version
    withdrawal   = the speaker withdraws an earlier statement without giving a replacement
    recollection = the speaker recalls what was said or was the case earlier ("I said in March that...")
    question     = only when the question itself is the evidence (e.g. CFO asks "is it 40 or 25?")
- statement: ONE self-contained sentence, pronouns resolved, names in full, past tense. It must be
  answerable from this unit alone.
- fact_key: a short canonical noun phrase naming the underlying question this claim answers, so that
  two claims about the same thing get the same key even in different words. Examples:
  "share of articles with shelf-life data populated", "pilot cohort store count",
  "bakery within fresh workstream scope", "OP_ID field in waste extract", "UAT sign-off scope",
  "twelve-month ROI expectation", "hosting region", "sub-processor count". Reuse keys within your file.
- value: the figure or categorical answer exactly as stated ("forty-eight percent", "25 stores",
  "yes, in scope", "dropped from next run"), or null.
- asserted_by: speaker/sender full name as given in the block header (Me/Them stay as Me/Them).
- hedged: true if maybe / roughly / about / I think / I believe / probably / "or so".
- secondhand: true if the speaker reports what someone else said or recalls a past statement.
- truncated: true if the sentence stops before the number or value is complete (e.g. "populated on
  forty-" or "the cost is roughly..."). Then value MUST be null. Never complete the number.
- corrects: for correction/withdrawal only: a short description of the earlier statement being
  corrected, including who said it and roughly when if stated. Else null.
- responds_to: for agreement/rejection only: what proposal is being answered and who made it. Else null.
- entities: list of people, systems, documents, stores, fields named in the claim.

What to extract: figures, dates, scope statements (what is in/out of a phase or workstream),
commitments, proposals and their acceptance or rejection, decisions, corrections, risks stated as
facts, who was told what. Skip greetings, filler ("Mm-hm", "Yeah"), and restatements inside the
same unit. A short turn like "Yes." that accepts a proposal in the previous turn IS an agreement:
extract it with responds_to filled in. Aim for completeness over brevity: every number in the
text should appear in some claim's value (or be marked truncated).
