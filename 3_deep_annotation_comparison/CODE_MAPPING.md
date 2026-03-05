# Code Mapping: CDP ↔ Gemini Behavioral Codes

## Overview

This document provides an explicit mapping between your legacy **CDP Score 1/2 annotation system** and Evey's **37 Gemini behavioral codes**, enabling systematic comparison and potential integration.

---

## Primary Mapping: Coordination Tiers

### CDP Score 1: Basic Coordination
**Definition:** Scaffolding, clarification, participation invitation, turn-taking

**Gemini Codes that Align:**

| Gemini Code | Category | Alignment Reasoning |
|---|---|---|
| `proposes_process` | Coordination & Decision Practices | Outlining structure, enabling team function |
| `invites_contribution` | Participation Dynamics | Explicitly enabling others to speak |
| `asks_clarifying_question` | Information Seeking | Scaffolding understanding |
| `elaborative_jump_in` | Participation Dynamics | Contributing to ongoing discussion thread |
| `uses_humor` | Relational Climate | Building interpersonal connection |
| `expresses_enthusiasm` | Relational Climate | Positive tone for collaboration |
| `audible_backchannel` | Engagement Signals | Showing active listening |
| `nod_count` | Engagement Signals | Visual confirmation/agreement |

**Approximate Coverage:** Score 1 utterances typically trigger 1-3 of these codes per utterance

---

### CDP Score 2: Advanced Coordination  
**Definition:** Decision-making, evaluation, commitment, idea crystallization

**Gemini Codes that Align:**

| Gemini Code | Category | Alignment Reasoning |
|---|---|---|
| `idea_quality` | Knowledge Sharing | Assessing idea merit (implicit decision-making) |
| `explicit_commitment_signal` | Commitment & Agreement | Explicit commitment = Score 2 in action |
| `shared_vision_indicator` | Relational Climate | Team converging on same mental model |
| `pronoun_shift_flag` | Pronoun Framing | "I" → "we" signals convergence (Score 2 territory) |
| `expertise_complementarity` | Complementarity Articulation | Recognizing how to combine strengths (coordination at decision level) |
| `skill_gap_identification` | Gap Identification | Recognizing gaps suggests decision framing |
| `risk_acknowledgment_with_enthusiasm` | Risk Management | Committing to actions despite uncertainty |
| `decision_crystallization_level` (1-4) | Overall Chunk Assessment | Direct parallel to aggregate Score 2 presence |

**Approximate Coverage:** Score 2 utterances typically trigger 1-2 of these codes, with higher decision_crystallization_level

---

## Secondary Mapping: Orthogonal Signals

### Signals Unique to Gemini (No CDP Equivalent)

These codes capture dimensions your CDP system doesn't measure:

#### Multimodal Engagement
| Code | What It Captures | Why CDP Misses It |
|------|---|---|
| `shared_affect` | Emotional synchrony (smiling, laughter together) | No video/audio tone in utterance-level analysis |
| `any_smile_other` | Non-verbal positive response | Text-based annotation only |
| `hesitation_flag` | Uncertainty in vocal delivery | Requires audio processing |
| `vocal_enthusiasm` (1-3 scale) | Energy level in speech | No prosody analysis in your system |
| `pace` (normal/slow/fast) | Delivery speed | Requires audio-level processing |
| `visible_off_screen_distraction` | Attention lapses | Requires video observation |
| `distracted_participant_count` | Collective attention | Requires video observation |

#### Linguistic Cohesion
| Code | What It Captures | Why CDP Misses It |
|------|---|---|
| `pronoun_shift_flag` | Individual vs. collective framing | Your system is utterance-content focused, not linguistic |
| `personal_disclosure` | Self-revelation building trust | Requires semantic understanding beyond coordination codes |
| `laughter_quality` (shared_humor, self-deprecating, etc.) | Social bonding texture | Beyond coordination/decision measurement |

#### Domain-Specific Integration
| Code | What It Captures | Why CDP Misses It |
|------|---|---|
| `cross_disciplinary_bridging` | Connecting ideas across domains | Requires domain knowledge; your Score 1/2 is domain-agnostic |
| `cross_disciplinary_bridging_speaker` | Who initiated the bridge | Would require role/expertise annotation |
| `cross_disciplinary_bridging_description` | Nature of the connection | Semantic, not coordination-focused |

#### Structural Awareness
| Code | What It Captures | Why CDP Misses It |
|------|---|---|
| `problem_specificity_level` (1-4) | How well-defined is the problem? | Your system scores utterances, not problem clarity |
| `ambition_level` | How ambitious are goals? | A psychological/cultural signal, not coordination-focused |
| `meeting_structure_quality` | How well-organized is the meeting? | Would need meta-level annotation |
| `screenshare_active` | Are artifacts being displayed? | Environmental/logistical, not behavioral |
| `artifact_type` & `artifact_interaction` | What's being shared/used? | Contextual, not behavioral |

#### Funding Awareness
| Code | What It Captures | Why CDP Misses It |
|------|---|---|
| `funding_awareness_signal` | Do they know funding requirements? | Domain-context specific |
| `funding_reference_description` | What funding is mentioned? | Semantic/content-level, not coordination |

#### Prior Knowledge
| Code | What It Captures | Why CDP Misses It |
|------|---|---|
| `prior_relationship_signal` | Do team members know each other? | Requires biographical knowledge |
| `prior_relationship_description` | What's the history? | Semantic context |

---

## Tertiary Mapping: Potential Synthesis Points

### How to Integrate Both Systems

#### 1. **Moment-Level Coding**
```
Utterance U at time T:
  ├─ CDP: Score 1 or Score 2 (utterance content)
  ├─ Gemini: Code category + subcode (behavioral signal)
  ├─ Engagement: nod_count, backchannel flag
  └─ Outcome: Does this move toward commitment? (yes/no/ambiguous)
```

#### 2. **Chunk-Level Aggregation**
```
Chunk C (time bin):
  ├─ CDP aggregate:
  │   ├─ Score 2 share: 0.48
  │   ├─ Entropy: 0.61
  │   └─ Speaker diversity (Gini): 0.32
  │
  ├─ Gemini aggregate:
  │   ├─ idea_trajectory: divergent
  │   ├─ decision_crystallization_level: 2
  │   ├─ engagement_level: 3
  │   └─ commitment_signals: 1
  │
  └─ Integrated signal:
      "Exploring options (divergent), with good engagement but low commitment (needs more convergence)"
```

#### 3. **Temporal Dynamics**
```
Sequence of chunks [C1, C2, C3, ..., C8]:
  
  CDP adds: entropy oscillations, speaker participation patterns
  Gemini adds: trajectory sequence, multimodal engagement arc
  
  Integration: "Team oscillated 3x between divergent exploration and ambiguous 
  transitions, with increasing engagement (nods, backchannels) but no explicit 
  commitment until chunk 7"
```

---

## Detailed Code Descriptions with Examples

### Coordination and Decision Practices (CDP-Aligned)

#### `proposes_process` ← CDP Score 1
**Gemini Definition:** Laying out steps, suggesting a method

**Example:** "Let's go around and have everyone introduce themselves"

**CDP Connection:** Process orientation enables basic coordination

---

#### `proposes_next_step` ← CDP Score 1/2 Border
**Gemini Definition:** Suggesting what to do next

**Example:** "Let's move to the whiteboard and sketch this out"

**CDP Connection:** Could be Score 1 (logistical) or Score 2 (decision-oriented)

---

#### `invites_contribution` ← CDP Score 1
**Gemini Definition:** Explicitly asking someone to speak

**Example:** "What do you think about that, Erica?"

**CDP Connection:** Participation facilitation = Score 1

---

#### `idea_quality` ← CDP Score 2
**Gemini Definition:** Assessing merit of an idea

**Example:** "That's interesting because it would leverage the existing infrastructure"

**CDP Connection:** Evaluating ideas = Score 2 territory

---

### Knowledge Sharing (Mostly CDP Score 1)

#### `shares_domain_knowledge` ← CDP Score 1
**Gemini Definition:** Providing expertise or background

**Example:** "In my lab, we use fly genetics for stem cell research"

**CDP Connection:** Sharing context = Score 1 (scaffolding understanding)

---

### Relational Climate (Mixed)

#### `uses_humor` ← CDP Score 1
**Gemini Definition:** Making a joke or lighthearted comment

**Example:** "She had the good idea, or bad idea, to move to Yale"

**CDP Connection:** Humor = interpersonal coordination (Score 1)

**Gemini Enhancement:** Captures *type* (shared_humor, self-deprecating, etc.)

---

#### `expresses_enthusiasm` ← CDP Score 1
**Gemini Definition:** Excitement, positive sentiment about topic/people

**Example:** "I love this topic and I'm so excited to meet you all"

**CDP Connection:** Positive relational signals = supporting Score 1 coordination

---

### Complementarity Articulation (Score 2 Border)

#### `expertise_complementarity` ← CDP Score 2
**Gemini Definition:** Explicitly recognizing how expertise combines

**Example:** "Hopefully I have complementary expertise to many of you"

**CDP Connection:** Recognizing how to integrate skills = Score 2 coordination

---

### Participation Dynamics (Mostly Score 1)

#### `elaborative_jump_in` ← CDP Score 1
**Gemini Definition:** Adding detail to others' points (elaborative interrupt)

**Example:** [While someone is explaining] "Right, and that's especially true for..."

**CDP Connection:** Collaborative building = Score 1

---

### Information Seeking (Score 1)

#### `asks_clarifying_question` ← CDP Score 1
**Gemini Definition:** Requesting clarification or more detail

**Example:** "When you say 'crowding,' do you mean molecular concentration?"

**CDP Connection:** Scaffolding understanding = Score 1

---

## Discrepancy Resolution Guide

### When Score 2 Is High But decision_crystallization_level Is Low

**Scenario:** Chunk has 60% Score 2 utterances but decision_crystallization_level = 1

**Possible Explanations:**
1. Team is *discussing* decisions without *making* them (exploring, not concluding)
2. Discussions are abstract/theoretical, not actionable
3. No explicit commitment signals despite evaluation language
4. Score 2 is about the *topic* (decisions in their field) not about *team decisions*

**Evey's Perspective:** "They're thinking hard, but not converging"

---

### When Score 2 Is Low But decision_crystallization_level Is High

**Scenario:** Chunk has 20% Score 2 but decision_crystallization_level = 3

**Possible Explanations:**
1. Team is converging through **nonverbal agreement** (nods, shared smiles)
2. Decisions are being made through **implicit consensus**, not explicit language
3. Gemini's multimodal signals are picking up commitment your utterance analysis misses
4. Task orientation without explicit decision language = coordinated action anyway

**Evey's Perspective:** "They're clearly on the same page, even if not saying it"

---

### When Entropy Is High But Trajectory Is "Procedural"

**Scenario:** Entropy = 0.72 (high mixing) but idea_trajectory = "procedural"

**Interpretation:** 
- Mixed coordination signals (good)
- But applied to *process/logistics*, not *idea exploration*
- Example: Discussing how to share materials (high Score 1/2 mix) but not engaging with ideas themselves

---

## Integration Recommendations for Your Code

### Option 1: Parallel Annotation (Recommended for validation)
Keep both systems independent, run joint analysis:
```python
def analyze_session(session_id):
    cdp_metrics = extract_cdp_metrics(session_id)      # Your system
    gemini_metrics = extract_gemini_metrics(session_id) # Evey's system
    
    # Align by time bins
    for i, cdp_bin in enumerate(cdp_metrics['binned']):
        alignment = {
            'cdp_score2_share': cdp_bin['score2_share'],
            'cdp_entropy': cdp_bin['entropy'],
            'gemini_trajectory': gemini_metrics[i]['idea_trajectory'],
            'gemini_decision_level': gemini_metrics[i]['decision_crystallization_level'],
            'match': score2_to_trajectory(cdp_bin['score2_share']) == gemini_metrics[i]['idea_trajectory']
        }
```

### Option 2: Blended Features (For outcome modeling)
```python
def extract_blended_features(session_id):
    return {
        # CDP features
        'entropy_mean': ...,
        'entropy_variance': ...,
        'gini_speaker_concentration': ...,
        'score2_trajectory': ...,
        
        # Gemini features  
        'trajectory_sequence': [...],
        'decision_crystallization_mean': ...,
        'commitment_signal_count': ...,
        'engagement_mean': ...,
        
        # Integration signals
        'entropy_decision_correlation': ...,
        'multimodal_congruence': ...,  # Do gestures match utterances?
    }
```

### Option 3: Hierarchical Annotation (For future data)
When coding new sessions:
1. **Start with Gemini chunk boundaries** (already identifying meaningful episodes)
2. **Within each chunk, apply CDP scoring** (fine-grained utterance-level)
3. **Aggregate both frameworks** per chunk
4. **Compare to baseline data** from this analysis

---

## Summary Matrix

| Dimension | CDP Captures | Gemini Captures | Integration Opportunity |
|-----------|---|---|---|
| **What** (idea content) | Score 1/2 type | Behavioral codes | Combine for rich semantics |
| **When** (temporal location) | Entropy dynamics | Chunk boundary | Trace idea evolution over time |
| **Who** (participation) | Speaker Gini | Multimodal engagement | Track both utterance & presence |
| **How engaged** (commitment) | Implicit in Score 2 | explicit_commitment_signal | Validate engagement metrics |
| **How stable** (oscillation) | Entropy variance | Trajectory stability | Assess team adaptability |

---

## Questions to Ask Evey

1. **Was decision_crystallization_level computed algorithmically or via annotation?**
   - If algorithmic: What's the formula? Can we harmonize with entropy?
   - If annotation: Can we compare annotator understanding of "crystallization" with our Score 2 definition?

2. **Are the 37 behavioral codes mutually exclusive within an utterance?**
   - Your system: one utterance → one max score (Score 1 or 2)
   - Gemini: one utterance → multiple codes possible?
   - This affects mapping strategy

3. **How were chunk boundaries determined?**
   - Time-based (fixed 10-min windows)?
   - Natural conversation break detection?
   - Human annotation?
   - If algorithm-based: Can we apply it to our data for better alignment?

4. **What's your intention for these codes going forward?**
   - Descriptive (document what happened)?
   - Predictive (predict outcomes)?
   - If predictive: Which codes are strongest outcome predictors?

5. **Would you be interested in a unified framework combining both approaches?**
   - Single annotation per utterance: Gemini behavioral code + CDP coordination score
   - Single aggregation per chunk: CDP metrics + Gemini assessment

---

*Mapping completed: March 5, 2026*
*Basis: 37 sessions, systematic comparison of two annotation frameworks*
