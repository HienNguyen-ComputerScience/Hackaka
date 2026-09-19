PRACTICE = {
    "P1": "What did the master-data assessment report as complete in September 2024? Give every figure and the document each comes from.",
    "P2": "What service levels were agreed for ordering, and in which meeting?",
    "P3": "Who proposed removing the operator ID field from the data extract, who agreed, and had it already been sent anywhere by then?",
    "P4": "Did Acme sign off UAT for the programme? Quote the scope of what was actually signed, and name who signed it.",
    "P5": "What proportion of articles had shelf-life data populated? Give every figure in the archive with its date and source, and say which one is current.",
    "P6": "Is bakery inside the fresh workstream? Show how the answer changed over time and what it is now.",
    "P8": "Find one thing in the archive that was agreed and then never done. Show the trail from the agreement to the present, and say who would have needed to notice.",
    "P9": "The weekly status reports say the nightly article extract completed with no errors. Is that true? Answer the question the reports are actually evidence for, and say what they are not evidence for.",
}

UNSEEN = [
    ("U01", "How many stores are in the pilot cohort, and did that number ever change?", True),
    ("U02", "What did the data protection officer ask RELEX to confirm, and was it closed?", True),
    ("U03", "When did the DC-2 feed fail during hypercare and who got paged?", True),
    ("U04", "What role does Meridian Consulting play in the bakery workstream?", True),
    ("U05", "Which field was excluded from the waste extract, and from what date?", True),
    ("U06", "Where is the RELEX environment hosted and how many sub-processors are there?", True),
    ("U07", "What return on investment did the CFO expect, and over what period?", True),
    ("U08", "Why was a personal note about surgery in the handover pack, and was it removed?", True),
    ("U09", "What went wrong with the three-wave rollout sequencing?", True),
    ("U10", "How was the case pack quantity gap closed and how long did it take?", True),
    ("U11", "What did the Q1 2026 waste reporting say about availability and waste?", True),
    ("U12", "Was a file size check adopted to detect feed failures?", True),
    ("U13", "What is the name of Acme's chief executive?", False),
    ("U14", "How many people work at Meridian Consulting?", False),   # phrased in archive vocabulary: no never-used word to catch it
    ("U15", "Which programming language is the DC-2 middleware written in?", False),
]

# Ten provenance questions for the judges, run against the full pre-deletion archive (never
# against the live data/, which may already reflect a deletion run earlier in this process or a
# prior session). See load_snapshot_answerer() / part_judge() below.
JUDGE_PROVENANCE = [
    ("J01", "Who first reported the DC-2 feed failure during hypercare? Give the speaker and the timestamp in the conversation, not just the document."),
    ("J02", "The operator ID field was excluded from the waste extract. Show me the single message where that was first put in writing, with its sender and sent date."),
    ("J03", "How many stores are in the pilot cohort? Give the figure and cite every document that states it, not just one."),
    ("J04", "What did Meridian Consulting charge for the bakery workstream?"),
    ("J05", "Quote the exact sentence in which the data protection officer states what they need confirmed."),
    ("J06", "How many sub-processors are there? If the archive disagrees with itself, cite both records and say which is which."),
    ("J07", "How many separate times was a file size check raised as a way to detect feed failures? Cite each occasion."),
    ("J08", "Was a data protection impact assessment ever carried out? If the archive does not say, explain how you determined that."),
    ("J09", "Did they switch all the shops on at the same time, or in batches? Show me where that is recorded."),
    ("J10", "What did the Q1 2026 waste reporting conclude? Name the report period inside the file it sits in, not the file."),
]

# Ten fixed questions for the Part 3 deletion diff. Near: topics whose chain has several authors
# and whose current statement belongs to the person Part 3 deletes (chosen the way P1's target
# is chosen: the top author of P1's statements). The chain survives the deletion with its head
# gone, so the renderer must say so and fall back to the superseded statement. Far: topics whose
# answer does not draw on that person at all. See part3() in test_full.py.
DIFF_NEAR = [
    ("D1", "How often did the DC-2 feed fail, and on which dates?"),
    ("D2", "What was in scope for the first fresh go-live?"),
    ("D3", "When was the fresh go/no-go decision scheduled?"),
    ("D4", "When was the pilot cohort reduced, and what was the impact?"),
    ("D5", "How many tonnes of fresh waste were avoided on the cohort?"),
]

# Attribution questions live in ../attribution_local.json (gitignored: they name corpus people as
# expected proposers/agreers, and a deleted person's name must not persist in versioned source).
# Row shape: [id, question, proposer, agreers, deciders, no_credit_before_proposal]. test_full.py
# skips the A-rows with a note when the file is absent. EVERY QUESTION MUST CONTAIN A WORD FROM
# answer.py's ATTRIB_WORDS (who / propos / suggest / agree / accept / sign / commit / decid / approv /
# confirm): without one, out["attribution"] is None and the row fails for a reason that has nothing
# to do with attribution.
ATTRIBUTION_FILE = "attribution_local.json"

# Three currency checks for Part 3, run on the full archive before the deletion. Each names the
# fact key of the chain it inspects: a stale chain must lead with its figure (a withdrawal is
# never the head), a never-true statement must stay visible with its NEVER TRUE line, and a
# withdrawal must carry its WITHDRAWN marker.
CURRENCY = [
    ("C1", "stale", "How many delivery waves does the fresh Phase 2 plan have?", "fresh delivery wave count"),
    ("C2", "never-true", "How many stores are in the pilot cohort?", "pilot cohort store count"),
    ("C3", "withdrawal", "Was a file size check adopted to detect feed failures?", "file size check as feed failure detection"),
]

DIFF_FAR = [
    ("D6", "What did the data protection officer ask RELEX to confirm?"),
    ("D7", "How many people work at Meridian Consulting?"),
    ("D8", "In the solution demo, was forecasting or replenishment walked through first?"),
    ("D9", "What is the total contract value of the Meridian Consulting agreement?"),
    ("D10", "Was RELEX allowed to put the value case in the statement of work?"),
]
