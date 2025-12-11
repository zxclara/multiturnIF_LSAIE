import random
import re
import time
from functools import lru_cache
from textwrap import dedent
from typing import Dict, List, Optional

from datasets import load_dataset, load_from_disk

import config

CATEGORY_ORDER = [
    "Reliable Version Editing",
    "Instruction Retention",
    "Inference Memory",
    "Self-Coherence",
]

CATEGORY_TO_AXIS = {
    "Reliable Version Editing": "RELIABLE_VERSION_EDITING",
    "Instruction Retention": "INSTRUCTION_RETENTION",
    "Inference Memory": "INFERENCE_MEMORY",
    "Self-Coherence": "SELF_COHERENCE",
}

CATEGORY_SHORT_DESC = {
    "Reliable Version Editing": "integrate evolving instructions over the dialogue without dropping earlier directives",
    "Instruction Retention": "uphold a specific instruction across all turns (tone/format/constraint/persona)",
    "Inference Memory": "recall specific details from earlier turns and reuse them correctly later",
    "Self-Coherence": "avoid contradictions across turns (numbers, facts, tone, policy, narrative consistency)",
}

CHALLENGE_DEFS = dedent(
    """
    Challenge types (long-dialogue instruction following):
    - Inference Memory: recall specific details from earlier turns and reuse them correctly later.
    - Instruction Retention: uphold a specific instruction across all turns (tone/format/constraint/persona).
    - Reliable Version Editing: integrate evolving instructions over the dialogue without dropping earlier directives.
    - Self-Coherence: avoid contradictions across turns (numbers, facts, tone, policy, narrative consistency).
    """
)

TRAP_GUIDE = dedent(
    """
    Trap design guidance (instruction-following oriented):
    - Embed the trap naturally in a realistic multi-turn dialogue; avoid adversarial or explicit “tests”.
    - The trap should be detectable in later assistant replies (trigger vs avoid is observable).
    - Align with long-form skills: instruction adherence, memory, coherence, evolving requirements.
    """
)

def build_planner_system(category_name: str) -> str:
    desc = CATEGORY_SHORT_DESC.get(category_name, "").strip()
    desc_line = f" (description: {desc})" if desc else ""
    return dedent(
        f"""
        You design {config.PLAN_TURNS}-turn instruction-following challenge plans for long dialogues.
        Goal: craft a realistic user-side question plan that subtly tests long-dialogue instruction-following; specifically, target Challenge Category: {category_name}{desc_line}.
        Output JSON only. Do not include explanations outside JSON.
        """
    ).strip()

CHALLENGE_TAXONOMY_SUMMARY = dedent(
    """
    Detailed challenge axes and common trap surfaces (from the paper appendix):
    - Reliable Version Editing: Technical (code/docs/SOPs), Writing & Content, Communication (email/memos/CS), Design & Presentation, Planning & Strategy, Professional & Career (CV/JD), Learning & Development, Research & Documentation, Customer & UX, Event & Content Planning.
    - Instruction Retention: Tone & Language (neutral/formal/specialized), Response Structure (limited answers/include specific element/consistent format), Grammar & Syntax (specific grammar/embed words), Behavioral Consistency (agree/objective persona), Challenging Formats (poetic/step-by-step).
    - Inference Memory: Personal Preference (allergies/tastes), Schedule & Time (dates/conflicts/recurring), Relationship Details, Location & Travel, Health & Fitness, Work & Project, Learning & Development, Hobbies & Interests, Shopping & Purchases, Entertainment & Media, Tasks & Reminders, Emotional State, Social & Cultural.
    - Self-Coherence: Numerical Consistency, Fact Retention, Policy & Regulation Consistency, Personal Information Consistency, Definition/Explanation Coherence, Recommendation Consistency, Instruction & Process Consistency, Mathematical Coherence, Contextual Coherence, Ethical & Moral Consistency, Story/Narrative Consistency, Opinion Consistency.
    """
)


def _clean_tex_text(text: str) -> str:
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


def clean_tex(path) -> str:
    return _clean_tex_text(path.read_text(encoding="utf-8"))


@lru_cache()
def load_taxonomy_categories() -> Dict[str, str]:
    """
    Return mapping {category_name: cleaned_text_section} parsed from taxonomy tex files.
    """
    categories: Dict[str, str] = {}
    for path in config.TAXONOMY_FILES:
        if not path.exists():
            continue
        cleaned = _clean_tex_text(path.read_text(encoding="utf-8"))
        for match in re.finditer(
            r"Challenge Category - ([^\n]+)\n(.*?)(?=Challenge Category - |\Z)",
            cleaned,
            flags=re.DOTALL,
        ):
            name = match.group(1).strip()
            name = re.sub(r"[}]+$", "", name).strip()
            body = match.group(2).strip()
            body = body.replace("\\newline", "\n")
            body = body.replace("newline", "\n")
            body = re.sub(r"\[\d+\.?\d*em\]", "", body)
            body = re.sub(r"\{\d+\}\{[^\}]+\}\{\{[^}]+\}\}", "", body)
            body = body.replace("[t]{}{", "")
            body = re.sub(r"\s*&\s*", " - ", body)
            body = body.replace("{", "").replace("}", "")
            body = re.sub(r"longtable[^\\n]*", "", body, flags=re.IGNORECASE)
            body = re.sub(r"arraybackslash", "", body, flags=re.IGNORECASE)
            body = "\n".join(
                line
                for line in body.splitlines()
                if "ngtable" not in line and ">p" not in line and line.strip() != "1l"
            )
            body = re.sub(r"\n{3,}", "\n\n", body)
            body = re.sub(r" +", " ", body)
            body = body.strip()
            categories[name] = body
    return categories


@lru_cache()
def load_taxonomy_text() -> str:
    parts = []
    for path in config.TAXONOMY_FILES:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8")
        segments = raw.split("\\midrule")
        if config.TAXONOMY_SAMPLE_RATIO >= 1.0:
            sampled_segments = segments
        else:
            k = max(1, int(len(segments) * config.TAXONOMY_SAMPLE_RATIO))
            sampled_segments = random.sample(segments, k)
        stitched = "\\midrule".join(sampled_segments)
        cleaned = _clean_tex_text(stitched)
        parts.append(f"=== {path.name} ===\n{cleaned}")
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
        Conversation (natural flow):
        {convo_text}
        Target trap (what to check later): {target_q}
        Pass criteria: {pass_criteria}
        """
    ).strip()


@lru_cache()
def load_fewshot_text(k: int = config.FEWSHOT_SAMPLES, axis: Optional[str] = None) -> str:
    local_path = config.HF_CACHE / "multichallenge"
    if local_path.exists():
        ds = load_from_disk(str(local_path))
    else:
        ds = load_dataset(
            config.FEWSHOT_DATASET_ID,
            split="train",
            cache_dir=str(config.HF_CACHE),
        )
    axis_filtered = ds
    if axis:
        axis_norm = axis.upper().replace(" ", "_")
        axis_filtered = ds.filter(lambda ex: ex.get("AXIS", "").upper() == axis_norm)
    if len(axis_filtered) == 0:
        return ""
    ds = axis_filtered
    total = len(ds)
    k = min(k, total)
    random.seed(int(time.time()))
    indices = random.sample(range(total), k)
    examples = [format_fewshot_example(ds[i], idx + 1) for idx, i in enumerate(indices)]
    stitched = "\n\n".join(examples)
    if len(stitched) > config.FEWSHOT_MAX_CHARS:
        stitched = stitched[: config.FEWSHOT_MAX_CHARS]
    return stitched


@lru_cache()
def load_planner_template_text() -> str:
    path = config.PLANNER_TEMPLATE_FILE
    if path.exists():
        return _clean_tex_text(path.read_text(encoding="utf-8"))
    return ""


def category_to_axis(category_name: str) -> str:
    if not category_name:
        return ""
    mapped = CATEGORY_TO_AXIS.get(category_name.strip())
    if mapped:
        return mapped
    return category_name.upper().replace(" ", "_").replace("-", "_")


def build_planner_user(
    seed_user: str,
    seed_assistant: str,
    plan_turns: int,
    category_name: str,
    category_details: str,
    fewshot_text: str,
    planner_template: str,
    switch_hint: bool = False,
) -> str:
    category_section = ""
    if category_name:
        details = category_details if category_details else "(no category details found in appendix)"
        category_section = dedent(
            f"""
            Target Challenge Category: {category_name}
            Sub-axes and guidance (from appendix):
            {details}

            Pick the most fitting subtopics in this category (describe them in English) and weave the user-side plan around them.
            """
        ).strip()

    fewshot_section = ""
    if config.USE_FEWSHOT and fewshot_text:
        fewshot_section = f"\nFew-shot examples (matching this challenge category):\n{fewshot_text}\n"

    template_section = ""
    if config.USE_AGENT_TEMPLATE and planner_template:
        template_section = f"\nReference planner prompt template (adapted from appendix, cleaned):\n{planner_template}\n"

    switch_section = ""
    if switch_hint:
        switch_section = dedent(
            """
            Optional twist (use only if it helps design a subtle challenge):
            You may briefly switch the dialogue mid-way to a few-shot-related topic for inspiration, then later return to the seed context.
            """
        ).strip()

    return dedent(
        f"""
        Dialogue can revolve around this seed context; you don't need to start with the exact same initial question. 
        The seed assistant reply is contextual/background flavor, not a ground-truth answer. 
        Feel free to open with a natural question inspired by the seed that best fits the chosen subtopics.

        Seed user message (context):
        {seed_user}

        Seed assistant reply (context, not authoritative):
        {seed_assistant}

        {category_section}

        {TRAP_GUIDE}

        {fewshot_section}
        {template_section}
        {switch_section}

        Requirements:
        - Use the target Challenge Category above; pick the most relevant subtopic(s) inside it and design around them.
        - Produce a realistic, context-linked user-side plan with {plan_turns} turns; questions must flow naturally (avoid rigid checklists), keep each question concise (<20 words).
        - Design a subtle trap: the plan should naturally surface the challenge without overt “tests”; keep trap/detection description concise.
        - Provide detection: how to tell the trap appears and whether it is triggered (observer criteria).

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
