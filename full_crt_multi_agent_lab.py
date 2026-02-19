"""Full CRT multi-agent lab simulation using a local Ollama endpoint.

Requires Ollama running on http://localhost:11434 with a compatible chat model.

Example:
    python full_crt_multi_agent_lab.py --rounds 3 --no-plot
    python full_crt_multi_agent_lab.py --save-plot artifacts/coherence.png
"""

from __future__ import annotations

import json
import argparse
from datetime import datetime
from pathlib import Path


# -------------------------
# CONFIGURATION
# -------------------------

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_ROUNDS = 5
DEFAULT_GOAL = (
    "Design a measurable entropy propagation model for adaptive multi-agent systems."
)

AGENTS = {
    "Architect": "You optimize systemic coherence and structural clarity.",
    "Critic": "You identify inconsistencies, weak assumptions, and missing quantification.",
    "Implementer": "You translate theory into executable and testable steps.",
}

agent_names = list(AGENTS.keys())
N = len(agent_names)

# Coupling matrix (directed influence strengths)
COUPLING_MATRIX = [
    [0.0, 0.6, 0.4],
    [0.5, 0.0, 0.7],
    [0.3, 0.6, 0.0],
]

# -------------------------
# LLM CALL
# -------------------------

def query_ollama(prompt: str, model: str, temperature: float, endpoint: str) -> str:
    import requests

    response = requests.post(
        endpoint,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CRT multi-agent entropy propagation simulation."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument(
        "--endpoint",
        default="http://localhost:11434/v1/chat/completions",
        help="Ollama-compatible chat completions endpoint",
    )
    parser.add_argument(
        "--save-plot",
        default="",
        help="Optional path to save coherence plot (recommended for headless environments)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting entirely",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON path for experiment history",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")

    coupling_matrix = COUPLING_MATRIX
    agent_confidence = [0.5] * N
    history = []
    coherence_over_time = []

    context_memory = ""

    for round_idx in range(args.rounds):
        print(f"\n===== ROUND {round_idx + 1} =====")

        responses = []
        entropies = []

        for i, agent in enumerate(agent_names):
            prompt = f"""
Role: {AGENTS[agent]}

System Goal:
{args.goal}

Shared Context:
{context_memory}

Your response must include:
- Core insight
- Concrete action
- Quantifiable metric
"""

            response = query_ollama(
                prompt,
                model=args.model,
                temperature=args.temperature,
                endpoint=args.endpoint,
            )
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

        # Entropy propagation
        propagated_entropy = [
            sum(coupling_matrix[row][col] * entropies[col] for col in range(N))
            for row in range(N)
        ]

        mean = sum(propagated_entropy) / len(propagated_entropy)
        variance = sum((x - mean) ** 2 for x in propagated_entropy) / len(propagated_entropy)
        coherence = 1 - variance
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
                "entropy": list(entropies),
                "propagated_entropy": list(propagated_entropy),
                "confidence": list(agent_confidence),
                "coherence": float(coherence),
            }
        )

        print(f"\n>>> System Coherence: {coherence:.4f}")
        print("Phases:", [phase(c) for c in agent_confidence])

    # -------------------------
    # VISUALIZATION
    # -------------------------

    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(range(1, args.rounds + 1), coherence_over_time)
        plt.axhline(0.6)
        plt.axhline(0.8)
        plt.title("CRT Coherence Over Time")
        plt.xlabel("Round")
        plt.ylabel("Coherence")
        if args.save_plot:
            plot_path = Path(args.save_plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=160, bbox_inches="tight")
            print(f"Saved plot to: {plot_path}")
        else:
            plt.show()

    # -------------------------
    # SAVE EXPERIMENT
    # -------------------------

    filename = args.output or f"crt_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nExperiment log saved as: {output_path}")


if __name__ == "__main__":
    main()
