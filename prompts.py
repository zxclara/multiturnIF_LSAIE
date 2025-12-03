from textwrap import dedent

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


def build_planner_user(seed_user: str, seed_assistant: str, plan_turns: int) -> str:
    return dedent(
        f"""
        Seed user message:
        {seed_user}

        Seed assistant reply (ground truth for context):
        {seed_assistant}

        {CHALLENGE_DEFS}

        {CHALLENGE_TAXONOMY_SUMMARY}

        {TRAP_GUIDE}

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
