"""CRT multi-agent entropy propagation lab.

Run against a local Ollama server or in mock mode for offline validation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import urllib.error
import urllib.request

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_ROUNDS = 5
DEFAULT_GOAL = "Design a measurable entropy propagation model for adaptive multi-agent systems."

AGENTS = {
    "Architect": "You optimize systemic coherence and structural clarity.",
    "Critic": "You identify inconsistencies, weak assumptions, and missing quantification.",
    "Implementer": "You translate theory into executable and testable steps.",
}

COUPLING_MATRIX = [
    [0.0, 0.6, 0.4],
    [0.5, 0.0, 0.7],
    [0.3, 0.6, 0.0],
]


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]


def variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((val - mean) ** 2 for val in values) / len(values)


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


def query_ollama(prompt: str, model: str, temperature: float, endpoint: str, timeout_s: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    return response_payload["choices"][0]["message"]["content"]


def mock_response(agent: str, round_idx: int, goal: str) -> str:
    return (
        f"Core insight: {agent} aligns entropy flow to system objective. "
        f"Concrete action: run experiment batch {round_idx + 1} with 10% perturbation. "
        "Quantifiable metric: measure coherence delta and response diversity percentage. "
        f"Goal link: {goal}"
    )


def run_simulation(
    rounds: int,
    model: str,
    temperature: float,
    goal: str,
    endpoint: str,
    timeout_s: int,
    mock: bool,
) -> tuple[list[dict], list[float]]:
    agent_names = list(AGENTS.keys())
    n_agents = len(agent_names)
    agent_confidence = [0.5] * n_agents

    history: list[dict] = []
    coherence_over_time: list[float] = []
    context_memory = ""

    for round_idx in range(rounds):
        print(f"\n===== ROUND {round_idx + 1} =====")

        responses: list[str] = []
        entropies: list[float] = []

        for i, agent in enumerate(agent_names):
            prompt = f"""
Role: {AGENTS[agent]}

System Goal:
{goal}

Shared Context:
{context_memory}

Your response must include:
- Core insight
- Concrete action
- Quantifiable metric
"""
            response = (
                mock_response(agent, round_idx, goal)
                if mock
                else query_ollama(prompt, model, temperature, endpoint, timeout_s)
            )
            responses.append(response)

            ent = compute_entropy(response)
            rew = reward_score(response)
            entropies.append(ent)

            agent_confidence[i] = min(agent_confidence[i] + 0.05 * rew, 1.0)

            print(f"\n--- {agent} ---")
            print(response[:600])
            print(f"Entropy: {ent:.3f} | Reward: {rew} | Confidence: {agent_confidence[i]:.2f}")

        propagated_entropy = matvec(COUPLING_MATRIX, entropies)
        coherence = float(1 - variance(propagated_entropy))
        coherence_over_time.append(coherence)

        weighted_context = ""
        for i, resp in enumerate(responses):
            weight = agent_confidence[i]
            weighted_context += f"\n[{agent_names[i]} | w={weight:.2f}]\n{resp}\n"
        context_memory = weighted_context

        history.append(
            {
                "round": round_idx + 1,
                "responses": responses,
                "entropy": entropies,
                "propagated_entropy": propagated_entropy,
                "confidence": agent_confidence,
                "coherence": coherence,
            }
        )

        print(f"\n>>> System Coherence: {coherence:.4f}")
        print("Phases:", [phase(c) for c in agent_confidence])

    return history, coherence_over_time


def main() -> None:
    parser = argparse.ArgumentParser(description="CRT multi-agent entropy propagation lab")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--endpoint", default="http://localhost:11434/v1/chat/completions")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--mock", action="store_true", help="Run without querying Ollama")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    history, coherence_over_time = run_simulation(
        rounds=args.rounds,
        model=args.model,
        temperature=args.temperature,
        goal=args.goal,
        endpoint=args.endpoint,
        timeout_s=args.timeout,
        mock=args.mock,
    )

    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(range(1, args.rounds + 1), coherence_over_time)
        plt.axhline(0.6)
        plt.axhline(0.8)
        plt.title("CRT Coherence Over Time")
        plt.xlabel("Round")
        plt.ylabel("Coherence")
        plt.tight_layout()
        plt.show()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    filename = outdir / f"crt_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with filename.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"\nExperiment log saved as: {filename}")


if __name__ == "__main__":
    main()
