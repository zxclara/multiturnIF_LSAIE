import random
import re
import time
from functools import lru_cache
from textwrap import dedent
from typing import List

from datasets import load_dataset, load_from_disk

import config

CHALLENGE_DEFS = dedent(
    """
    Challenge types (from Multi-Challenge paper):
    - Inference Memory: test recall of specific details from earlier turns and use them correctly later.
    - Instruction Retention: uphold a specific instruction for the entire conversation (tone, format, constraint, persona).
    - Reliable Version Editing: integrate evolving instructions over the dialogue without dropping or contradicting earlier directives.
    - Self-Coherence: avoid contradictions across turns (numbers, facts, tone, policy, narrative consistency).
    """
)

TRAP_GUIDE = dedent(
    """
    Trap design guidance:
    - Set up an explicit target check (the trap) that appears naturally in the planned turns.
    - Make the trap observable (define what a triggered vs avoided response looks like).
    - Keep the conversation realistic; no gaslighting or contrived failures.
    """
)

PLANNER_SYSTEM = dedent(
    """
    You are a planner designing a 15-turn, trap-aware user question plan to test a model.
    Role: vulnerability researcher posing as an innocent user.
    Output JSON only. Do not include explanations outside JSON.
    """
)

CHALLENGE_TAXONOMY_SUMMARY = dedent(
    """
    Detailed challenge axes and common trap surfaces (from the paper appendix):
    - Reliable Version Editing: Technical (code/docs/SOPs), Writing & Content, Communication (email/memos/CS), Design & Presentation, Planning & Strategy, Professional & Career (CV/JD), Learning & Development, Research & Documentation, Customer & UX, Event & Content Planning.
    - Instruction Retention: Tone & Language (neutral/formal/specialized), Response Structure (limited answers/include specific element/consistent format), Grammar & Syntax (specific grammar/embed words), Behavioral Consistency (agree/objective persona), Challenging Formats (poetic/step-by-step).
    - Inference Memory: Personal Preference (allergies/tastes), Schedule & Time (dates/conflicts/recurring), Relationship Details, Location & Travel, Health & Fitness, Work & Project, Learning & Development, Hobbies & Interests, Shopping & Purchases, Entertainment & Media, Tasks & Reminders, Emotional State, Social & Cultural.
    - Self-Coherence: Numerical Consistency, Fact Retention, Policy & Regulation Consistency, Personal Information Consistency, Definition/Explanation Coherence, Recommendation Consistency, Instruction & Process Consistency, Mathematical Coherence, Contextual Coherence, Ethical & Moral Consistency, Story/Narrative Consistency, Opinion Consistency.
    """
)


def clean_tex(path) -> str:
    text = path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("%"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = text.replace("\\\\", "\n")
    text = re.sub(r"\\(toprule|midrule|bottomrule|hline|cline\\{[^}]+\\})", "", text)
    text = re.sub(r"\\(begin|end)\\{[^}]+\\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\\*", "", text)
    text = re.sub(r"\\textbf\\{([^}]+)\\}", r"\\1", text)
    text = re.sub(r"\\label\\{[^}]+\\}", "", text)
    text = re.sub(r"\\caption\\{[^}]+\\}", "", text)
    text = re.sub(r"\\{", "{", text)
    text = re.sub(r"\\}", "}", text)
    text = re.sub(r"\\$", "", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@lru_cache()
def load_taxonomy_text() -> str:
    parts = []
    for path in config.TAXONOMY_FILES:
        if path.exists():
            parts.append(f"=== {path.name} ===\n{clean_tex(path)}")
    return "\n\n".join(parts)


def format_fewshot_example(example: dict, idx: int) -> str:
    convo_lines = []
    for turn in example.get("CONVERSATION", []):
        role = turn.get("role", "").upper()
        content = turn.get("content", "")
        convo_lines.append(f"{role}: {content}")
    convo_text = "\n".join(convo_lines)
    target_q = example.get("TARGET_QUESTION", "")
    pass_criteria = example.get("PASS_CRITERIA", "")
    axis = example.get("AXIS", "")
    return dedent(
        f"""
        Example {idx} (AXIS={axis}):
        Conversation, please note how the user questions flow natrually:
        {convo_text}
        Target trap: {target_q}
        passing criteria for trap: {pass_criteria}
        """
    ).strip()


@lru_cache()
def load_fewshot_text(k: int = config.FEWSHOT_SAMPLES) -> str:
    local_path = config.HF_CACHE / "multichallenge"
    if local_path.exists():
        ds = load_from_disk(str(local_path))
    else:
        ds = load_dataset(
            config.FEWSHOT_DATASET_ID,
            split="train",
            cache_dir=str(config.HF_CACHE),
        )
    total = len(ds)
    k = min(k, total)
    random.seed(int(time.time()))
    indices = random.sample(range(total), k)
    examples = [format_fewshot_example(ds[i], idx + 1) for idx, i in enumerate(indices)]
    return "\n\n".join(examples)


def build_planner_user(
    seed_user: str,
    seed_assistant: str,
    plan_turns: int,
    taxonomy_text: str,
    fewshot_text: str,
) -> str:
    taxonomy_section = ""
    if config.USE_TAXONOMY and taxonomy_text:
        taxonomy_section = f"\nFull taxonomy and trap surfaces (verbatim from appendix):\n{taxonomy_text}\n"

    fewshot_section = ""
    if config.USE_FEWSHOT and fewshot_text:
        fewshot_section = f"\nFew-shot examples (from nmayorga7/multichallenge):\n{fewshot_text}\n"

    return dedent(
        f"""
        Seed user message:
        {seed_user}

        Seed assistant reply (ground truth for context):
        {seed_assistant}

        {CHALLENGE_DEFS}

        {TRAP_GUIDE}

        {taxonomy_section}
        {fewshot_section}

        Requirements:
        - Select exactly one challenge type (choose the best fit for this seed).
        - Design a trap-aware conversation plan with {plan_turns} user turns (questions/prompts).
        - The plan must be realistic, context-linked, and flow naturally.
        - Include how to detect if the trap appears and how to check whether it is triggered.

        Return a JSON object with keys:
        {{
          "challenge_type": "<one of: Inference Memory | Instruction Retention | Reliable Version Editing | Self-Coherence>",
          "trap_summary": "<brief description of the trap and success/fail criteria>",
          "questions": ["turn1 user message", ..., "turn{plan_turns} user message"],
          "detection": "<how to check trap presence and triggering>",
          "notes": "<other evaluation details if any>"
        }}
        """
    ).strip()


RESPONDER_SYSTEM = dedent(
    """
    You are a helpful assistant. Use the provided seed answer as background context, but respond freshly to each user message.
    Stay coherent and natural.
    """
)


def build_responder_system(seed_assistant: str) -> str:
    return dedent(
        f"""
        {RESPONDER_SYSTEM}
        Ground truth answer for reference:
        {seed_assistant}
        """
    ).strip()


def build_evaluator_prompt(plan_text: str, trap_summary: str, detection: str, messages: list) -> str:
    conversation = []
    for turn in messages:
        role = turn.get("role", "")
        content = turn.get("content", "")
        conversation.append(f"{role.upper()}: {content}")
    convo_text = "\n".join(conversation)

    return dedent(
        f"""
        You are an evaluator for multi-challenge dialogues.
        Plan/trap summary: {trap_summary}
        Detection instructions: {detection}

        Conversation:
        {convo_text}

        Return JSON with:
        {{
          "trap_present": true/false,
          "trap_triggered": true/false,
          "naturalness": <score 1-10>,
          "authenticity": <score 1-10>,
          "safety": <score 1-10>,
          "accuracy": <score 1-10>,
          "comments": "<short rationale>"
        }}
        If the trap never appears, set trap_present=false and trap_triggered=false and still score the trajectory quality.
        """
    ).strip()
