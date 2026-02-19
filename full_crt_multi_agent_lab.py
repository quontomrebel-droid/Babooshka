"""Full CRT multi-agent lab simulation using a local Ollama endpoint.

Requires Ollama running on http://localhost:11434 with a compatible chat model.
"""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import requests

# -------------------------
# CONFIGURATION
# -------------------------

MODEL = "llama3.1:8b"
TEMPERATURE = 0.4
ROUNDS = 5
GOAL = "Design a measurable entropy propagation model for adaptive multi-agent systems."

AGENTS = {
    "Architect": "You optimize systemic coherence and structural clarity.",
    "Critic": "You identify inconsistencies, weak assumptions, and missing quantification.",
    "Implementer": "You translate theory into executable and testable steps.",
}

agent_names = list(AGENTS.keys())
N = len(agent_names)

# Coupling matrix (directed influence strengths)
coupling_matrix = np.array(
    [
        [0.0, 0.6, 0.4],
        [0.5, 0.0, 0.7],
        [0.3, 0.6, 0.0],
    ]
)

# Initial confidence
agent_confidence = np.ones(N) * 0.5

# History logs
history = []
coherence_over_time = []


# -------------------------
# LLM CALL
# -------------------------

def query_ollama(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# -------------------------
# METRICS
# -------------------------

def compute_entropy(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)



def reward_score(response: str) -> int:
    score = 0
    lower = response.lower()
    if "experiment" in lower:
        score += 1
    if "%" in response:
        score += 1
    if "measure" in lower:
        score += 1
    return score



def phase(conf: float) -> str:
    if conf < 0.4:
        return "Fragmented"
    if conf < 0.6:
        return "Converging"
    return "Coherent"


# -------------------------
# SIMULATION LOOP
# -------------------------

def main() -> None:
    context_memory = ""

    for round_idx in range(ROUNDS):
        print(f"\n===== ROUND {round_idx + 1} =====")

        responses = []
        entropies = []

        for i, agent in enumerate(agent_names):
            prompt = f"""
Role: {AGENTS[agent]}

System Goal:
{GOAL}

Shared Context:
{context_memory}

Your response must include:
- Core insight
- Concrete action
- Quantifiable metric
"""

            response = query_ollama(prompt)
            responses.append(response)

            ent = compute_entropy(response)
            rew = reward_score(response)
            entropies.append(ent)

            # Reinforcement update
            agent_confidence[i] += 0.05 * rew
            agent_confidence[i] = min(agent_confidence[i], 1.0)

            print(f"\n--- {agent} ---")
            print(response[:600])
            print(
                f"Entropy: {ent:.3f} | Reward: {rew} | Confidence: {agent_confidence[i]:.2f}"
            )

        entropies = np.array(entropies)

        # Entropy propagation
        propagated_entropy = coupling_matrix @ entropies

        coherence = 1 - np.var(propagated_entropy)
        coherence_over_time.append(float(coherence))

        # Update context memory (weighted by confidence)
        weighted_context = ""
        for i, resp in enumerate(responses):
            weight = agent_confidence[i]
            weighted_context += f"\n[{agent_names[i]} | w={weight:.2f}]\n{resp}\n"

        context_memory = weighted_context

        history.append(
            {
                "round": round_idx + 1,
                "responses": responses,
                "entropy": entropies.tolist(),
                "propagated_entropy": propagated_entropy.tolist(),
                "confidence": agent_confidence.tolist(),
                "coherence": float(coherence),
            }
        )

        print(f"\n>>> System Coherence: {coherence:.4f}")
        print("Phases:", [phase(c) for c in agent_confidence])

    # -------------------------
    # VISUALIZATION
    # -------------------------

    plt.figure()
    plt.plot(range(1, ROUNDS + 1), coherence_over_time)
    plt.axhline(0.6)
    plt.axhline(0.8)
    plt.title("CRT Coherence Over Time")
    plt.xlabel("Round")
    plt.ylabel("Coherence")
    plt.show()

    # -------------------------
    # SAVE EXPERIMENT
    # -------------------------

    filename = f"crt_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nExperiment log saved as: {filename}")


if __name__ == "__main__":
    main()
