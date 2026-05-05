"""Prompt templates for the PHT framework coding task.

This module was rewritten in April 2026 to address specific quality problems
identified by Shelby Hagemann's analysis of an earlier run (notably: high
confidence on wrong answers, heavy use of the forced-choice fallback, poor
agreement with human consensus on dimensions 1, 5, 6, 7).

The rewrite applies techniques from SICCS CS 599 prompt engineering lectures:

  - Four-part prompt structure (lec. 14): instruction, context, input, output
  - Role prompting (lec. 16): cast the model as a trained PHT coder, not a
    generic "research assistant"
  - Framework context: the model is told what PHT is and what each dimension
    means, instead of having to infer from parenthetical examples
  - Chain-of-thought (lec. 15): reasoning is produced BEFORE the answer so
    the model commits to a line of reasoning rather than rationalizing a
    snap judgment
  - Calibrated confidence: anchor definitions for 0.3 / 0.6 / 0.9 so scores
    are comparable across models and papers
  - One dimension-specific few-shot example (lec. 15): uses Miller 2007
    "Finger Dance," where all four models and the human consensus agreed
  - Explicit Not-Applicable path: some papers do not describe a game or
    play system. Forcing them onto a 1-5 scale manufactures noise. The
    prompt now lets the model say "Not Applicable" honestly.
"""
from __future__ import annotations

from typing import List

from ingest.chunker import TextChunk


# ---------------------------------------------------------------------------
# Framework context block — shared by every prompt call.
# Keeps the role + framework definition consistent. Placed first so it acts
# like a lightweight system prompt within the user message.
# ---------------------------------------------------------------------------
_ROLE_AND_FRAMEWORK = """You are a trained coder for a systematic literature review on Playful Health Technology (PHT). Your job is to read excerpts from an accessibility-focused research paper and assign it a score on one PHT framework dimension.

ABOUT THE PHT FRAMEWORK
The PHT framework (Duval, 2022) characterizes games and play systems in accessibility research across eight design dimensions. Each dimension places the system on a 1-to-5 spectrum. The score describes the DESIGNED SYSTEM or STUDY ARTIFACT the paper reports on — not the paper's topic, not the authors' intent, not the participants.

CODING PRINCIPLES
1. Read the evidence first. Only code what the paper actually describes.
2. If the paper does not describe a game or play system (e.g. it is a survey,
   interview study, or tool evaluation with no play artifact), answer
   "Not Applicable". Do not guess.
3. If the paper describes a system but the text you see does not cover this
   specific dimension, answer "Unknown". Do not guess.
4. Some papers describe multiple systems. Code the main artifact the paper
   centers on. Note this in reasoning if ambiguous.
5. Be cautious about the parenthetical examples in the question. They
   illustrate the option but do not define it. A paper can land on option (1)
   without literally re-appropriating an entertainment game.
"""


# ---------------------------------------------------------------------------
# Calibration anchors for the confidence field.
# Without these, confidence scores are not comparable across questions or
# models. This follows Shelby's finding that uncalibrated confidence stays
# artificially high even on wrong answers.
# ---------------------------------------------------------------------------
_CONFIDENCE_ANCHORS = """CONFIDENCE CALIBRATION (important)
Use this scale for the confidence field:
  0.90 - 1.00   The paper states this explicitly or describes the mechanic in
                detail. Multiple passages in context corroborate the answer.
  0.70 - 0.89   The paper implies this clearly through described mechanics,
                but does not state it directly. One or more passages support.
  0.50 - 0.69   Reasonable inference from partial evidence. The text is
                consistent with the answer but not conclusive.
  0.30 - 0.49   Weak inference. You are extrapolating from limited cues.
                Consider answering "Unknown" instead.
  0.00 - 0.29   You are essentially guessing. Answer "Unknown" or
                "Not Applicable" in this case.
"""


# ---------------------------------------------------------------------------
# A single dimension-specific worked example.
# Using miller07 ("Finger Dance: A sound game for blind people") because
# the human consensus and all four models agreed on dimensions 2, 3, 4, 8.
# For each question we show the model the relevant reasoning pattern.
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES: dict[int, str] = {
    # Dimension 1: Fun-First vs. Utility-First
    0: """WORKED EXAMPLE
Paper excerpt: "We designed Finger Dance, a rhythm game for blind users, as an
enjoyable sound-based experience. Our primary goal was to provide entertainment
comparable to sighted players' experience with Dance Dance Revolution, while
ensuring the audio design made timing cues fully accessible."

Reasoning: The stated primary goal is entertainment comparable to DDR. Accessibility
is a design constraint, not a utilitarian purpose like therapy or training. This
fits "totally fun first" (entertainment goal, re-appropriated from an existing
entertainment game concept).

Correct answer: 1
Confidence: 0.9 (primary goal stated explicitly)
""",
    # Dimension 2: Play vs. Game
    1: """WORKED EXAMPLE
Paper excerpt: "Players step on arrow pads in time with the beat. Correct steps
earn points; missed steps lose points. Score thresholds unlock subsequent songs.
The game ends when a song completes."

Reasoning: Rigid rule set with clear win/loss conditions, scoring, and
progression gates. Classic rigid-rule video game structure.

Correct answer: 5
Confidence: 0.95 (rules stated explicitly)
""",
    # Dimension 3: Skill vs. Chance
    2: """WORKED EXAMPLE
Paper excerpt: "Performance depends entirely on the player's ability to
recognize audio patterns and respond within the timing window. No randomization
or dice mechanics are used."

Reasoning: Pure skill. Paper explicitly rules out chance mechanics.

Correct answer: 1
Confidence: 0.95 (explicitly stated)
""",
    # Dimension 4: Solo vs. Social
    3: """WORKED EXAMPLE
Paper excerpt: "Finger Dance is played by a single user at one keyboard. There
is no multiplayer mode, leaderboard, or co-play affordance."

Reasoning: Solo by design. Paper explicitly excludes multiplayer.

Correct answer: 1
Confidence: 0.95 (explicitly stated)
""",
    # Dimension 5: Sequential vs. Simultaneous
    4: """WORKED EXAMPLE
Paper excerpt: "Cue sounds play at fixed rhythm intervals. The player must
respond before the next beat. Actions happen in real time as the song plays."

Reasoning: Real-time continuous action, not turn-based. The "phases" version
would apply if the game had e.g. a setup phase and a performance phase; this
has continuous play. Landing on 5 (entirely simultaneous / real-time).

Correct answer: 5
Confidence: 0.85 (clear from mechanic description)
""",
    # Dimension 6: Synchronous vs. Asynchronous
    5: """WORKED EXAMPLE
Paper excerpt: "The game requires immediate response to audio cues. There is
no save-and-resume feature or asynchronous play mode."

Reasoning: Entirely synchronous — play requires continuous real-time presence.
No asynchronous affordances described.

Correct answer: 1
Confidence: 0.9 (explicitly stated)
""",
    # Dimension 7: Competitive vs. Collaborative
    6: """WORKED EXAMPLE
Paper excerpt: "Players compete against their own previous high scores. Leaderboard
functionality was tested in pilot but removed from the final version."

Reasoning: Competitive framing — self-competition is still competitive in this
framework. No collaboration mechanics. Landing on 1 (entirely competitive).

Correct answer: 1
Confidence: 0.8 (described, but self-competition is the edge case)
""",
    # Dimension 8: Symmetrical vs. Asymmetrical
    7: """WORKED EXAMPLE
Paper excerpt: "The single player performs the same action set throughout.
There are no role differences because the game is single-player."

Reasoning: Symmetry is about whether different players have different abilities.
In a solo game, the player has only one role, so this maps to fully symmetrical.

Correct answer: 1
Confidence: 0.85 (follows from solo + uniform mechanics)
""",
}


# ---------------------------------------------------------------------------
# Output schema — unchanged in structure, but note that reasoning now comes
# BEFORE the answer in the JSON. The ordering here matters: the model is
# asked to produce the keys in order, which nudges chain-of-thought.
# ---------------------------------------------------------------------------
_OUTPUT_SCHEMA = """OUTPUT FORMAT
Return STRICT JSON only. No markdown fences, no commentary outside the JSON.
Produce fields in this exact order — the "reasoning" field must come first:

{
  "reasoning": "2-4 sentences. State what you found in the evidence, then explain which option it matches and why. Work through the logic before naming the answer.",
  "evidence": ["chunk_id_1", "chunk_id_2"],
  "answer": 1 | 2 | 3 | 4 | 5 | "Unknown" | "Not Applicable",
  "confidence": 0.00 to 1.00 (see calibration scale above)
}

"Unknown" means the paper likely has an answer but the excerpts shown do not
reveal it. "Not Applicable" means the paper does not describe a game or play
system at all.
"""


def _render_context(chunks: List[TextChunk]) -> str:
    """Render retrieved chunks as the Paper Context block."""
    lines = []
    for chunk in chunks:
        lines.append(
            f"[CHUNK_ID={chunk.chunk_id} | PAGES={chunk.page_start}-{chunk.page_end}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(lines)


def build_prompt(question: str, chunks: List[TextChunk], question_index: int | None = None) -> str:
    """Build the full prompt for one question + paper context.

    If question_index is provided, a dimension-specific few-shot example is
    included. If not (legacy callers), the prompt still works — it just falls
    back to zero-shot with the framework context and calibration.
    """
    parts = [
        _ROLE_AND_FRAMEWORK,
        _CONFIDENCE_ANCHORS,
    ]

    if question_index is not None and question_index in _FEW_SHOT_EXAMPLES:
        parts.append(_FEW_SHOT_EXAMPLES[question_index])

    parts.append(_OUTPUT_SCHEMA)

    parts.append("RUBRIC QUESTION\n" + question.strip())

    parts.append("PAPER CONTEXT (retrieved excerpts)\n" + _render_context(chunks))

    # Final nudge to produce the answer in the required order.
    parts.append(
        "Now produce the JSON response. Remember: reasoning field first, then "
        "evidence, then answer, then confidence. If the evidence is weak, prefer "
        '"Unknown" over guessing. If the paper does not describe a game or play '
        'system, answer "Not Applicable".'
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# NOTE: build_forced_choice_prompt has been deliberately removed.
#
# Shelby Hagemann's analysis (April 2026) showed that forced_choice decisions
# correlate with inaccuracy: models asked to guess when evidence is absent
# produce plausible-but-wrong answers with inflated confidence. This is the
# exact hallucination pattern warned about in the CS 599 LLM lecture (lec. 22).
#
# The pipeline now treats "Unknown" and "Not Applicable" as valid, honest
# answers. They are surfaced in the UI rather than hidden behind a forced guess.
# ---------------------------------------------------------------------------
