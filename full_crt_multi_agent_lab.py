"""Full CRT multi-agent lab simulation using a local Ollama endpoint.

Requires Ollama running on http://localhost:11434 with a compatible chat model.

Example:
    python full_crt_multi_agent_lab.py --rounds 3 --no-plot
    python full_crt_multi_agent_lab.py --save-plot artifacts/coherence.png
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib import error, request

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

AGENT_NAMES = list(AGENTS.keys())
N = len(AGENT_NAMES)

COUPLING_MATRIX = [
    [0.0, 0.6, 0.4],
    [0.5, 0.0, 0.7],
    [0.3, 0.6, 0.0],
]


# -------------------------
# LLM CALL
# -------------------------

def query_ollama(prompt: str, model: str, temperature: float, endpoint: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama endpoint. Ensure Ollama is running and endpoint is correct. "
            f"endpoint={endpoint}"
        ) from exc

    try:
        content = json.loads(raw)
        return content["choices"][0]["message"]["content"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Received an unexpected response format from the LLM endpoint."
        ) from exc


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
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting entirely")
    parser.add_argument(
        "--output", default="", help="Optional output JSON path for experiment history"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run without Ollama by generating deterministic mock agent responses",
    )
    parser.add_argument(
        "--fallback-mock",
        action="store_true",
        help="If live Ollama call fails, continue run with mock responses",
    )
    return parser.parse_args()


def mock_response(agent: str, round_idx: int, goal: str) -> str:
    return (
        f"Core insight: {agent} aligns with the system objective by reducing uncertainty around '{goal}'.\n"
        f"Concrete action: run experiment batch R{round_idx + 1} with ablations and cross-agent review.\n"
        "Quantifiable metric: measure coherence gain target of 12% and entropy spread below 0.08."
    )


def main() -> None:
    args = parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")

    agent_confidence = [0.5] * N
    history = []
    coherence_over_time = []
    context_memory = ""
    fallback_warned = False

    for round_idx in range(args.rounds):
        print(f"\n===== ROUND {round_idx + 1} =====")
        responses = []
        entropies = []

        for i, agent in enumerate(AGENT_NAMES):
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

            if args.mock:
                response = mock_response(agent=agent, round_idx=round_idx, goal=args.goal)
            else:
                try:
                    response = query_ollama(
                        prompt,
                        model=args.model,
                        temperature=args.temperature,
                        endpoint=args.endpoint,
                    )
                except RuntimeError:
                    if not args.fallback_mock:
                        raise
                    if not fallback_warned:
                        print(
                            "WARNING: Live Ollama call failed; switching to mock responses for this run.",
                            file=sys.stderr,
                        )
                        fallback_warned = True
                    response = mock_response(agent=agent, round_idx=round_idx, goal=args.goal)
            responses.append(response)

            ent = compute_entropy(response)
            rew = reward_score(response)
            entropies.append(ent)

            agent_confidence[i] = min(agent_confidence[i] + 0.05 * rew, 1.0)

            print(f"\n--- {agent} ---")
            print(response[:600])
            print(
                f"Entropy: {ent:.3f} | Reward: {rew} | Confidence: {agent_confidence[i]:.2f}"
            )

        propagated_entropy = [
            sum(COUPLING_MATRIX[row][col] * entropies[col] for col in range(N))
            for row in range(N)
        ]

        mean = sum(propagated_entropy) / len(propagated_entropy)
        variance = sum((x - mean) ** 2 for x in propagated_entropy) / len(propagated_entropy)
        coherence = 1 - variance
        coherence_over_time.append(float(coherence))

        weighted_context = ""
        for i, resp in enumerate(responses):
            weighted_context += f"\n[{AGENT_NAMES[i]} | w={agent_confidence[i]:.2f}]\n{resp}\n"
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

    filename = args.output or f"crt_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nExperiment log saved as: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
