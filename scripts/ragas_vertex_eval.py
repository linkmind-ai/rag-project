from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from google import genai
from ragas import evaluate
from ragas.embeddings import GoogleEmbeddings
from ragas.llms import llm_factory
from ragas.run_config import RunConfig

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)

from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    ResponseRelevancy as AnswerRelevancy,
)

try:
    from ragas.dataset_schema import EvaluationDataset
except ImportError:
    EvaluationDataset = None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_correctness",
    "answer_relevancy",
)
PERSONA_KEYS = (
    "preferred_topics",
    "avoid_topics",
    "response_style",
    "factuality_bias",
    "explicit_notes",
)


class VertexGenAIEmbeddingsAdapter(GoogleEmbeddings):
    """Expose the query/document embedding interface expected by ragas metrics."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the local RAG API with ragas using Vertex AI Gemini."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.getenv("RAGAS_EVAL_LIMIT", "10")),
        help="Number of golden-set samples to evaluate.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("RAGAS_HTTP_TIMEOUT", "60")),
        help="HTTP timeout in seconds for each /query call.",
    )
    parser.add_argument(
        "--golden-set",
        default=os.getenv("RAGAS_GOLDEN_SET", "tests/golden_set_100.json"),
        help="Path to the golden-set JSON file.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("RAGAS_QUERY_API_URL", "http://127.0.0.1:8000/query"),
        help="Query API endpoint for the LangGraph app.",
    )
    parser.add_argument(
        "--output-prefix",
        default=os.getenv(
            "RAGAS_OUTPUT_PREFIX", ".benchmarks/ragas_vertex_eval_golden_set_100"
        ),
        help="Output file prefix without extension.",
    )
    parser.add_argument(
        "--cache-path",
        default=os.getenv("RAGAS_CACHE_PATH", ""),
        help="Optional path to cache collected query results as JSON.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip /query collection and evaluate from the cache file only.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=(
            "Comma-separated metric names. "
            f"Available: {', '.join(DEFAULT_METRICS)}"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("RAGAS_MAX_WORKERS", "2")),
        help="Maximum concurrent ragas worker tasks.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("RAGAS_BATCH_SIZE", "2")),
        help="Optional ragas batch size.",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=int(os.getenv("RAGAS_TIMEOUT", "300")),
        help="Per-metric ragas timeout in seconds.",
    )
    parser.add_argument(
        "--ragas-max-retries",
        type=int,
        default=int(os.getenv("RAGAS_MAX_RETRIES", "12")),
        help="Retry count for ragas metric calls.",
    )
    parser.add_argument(
        "--ragas-max-wait",
        type=int,
        default=int(os.getenv("RAGAS_MAX_WAIT", "90")),
        help="Max backoff wait in seconds for ragas retries.",
    )
    parser.add_argument(
        "--persona-file",
        default=os.getenv("RAGAS_PERSONA_FILE", ""),
        help="Optional JSON file containing a session persona/profile.",
    )
    return parser.parse_args()


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def build_google_client() -> tuple[Any, bool, str | None, str | None]:
    api_key = env("VERTEX_API_KEY")
    project_id = env("GOOGLE_CLOUD_PROJECT")
    location = env("GOOGLE_CLOUD_LOCATION", "global")

    if api_key:
        client = genai.Client(vertexai=True, api_key=api_key)
        return client, True, None, None

    if project_id:
        client = genai.Client(vertexai=True, project=project_id, location=location)
        return client, True, project_id, location

    raise RuntimeError(
        "Set VERTEX_API_KEY for Vertex AI express mode, or set "
        "GOOGLE_CLOUD_PROJECT plus GOOGLE_CLOUD_LOCATION for full Vertex AI."
    )


def parse_metric_names(raw_metrics: str) -> list[str]:
    selected = [name.strip() for name in raw_metrics.split(",") if name.strip()]
    if not selected:
        raise RuntimeError("At least one metric must be selected.")

    unknown = [name for name in selected if name not in DEFAULT_METRICS]
    if unknown:
        raise RuntimeError(
            "Unknown metrics: "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(DEFAULT_METRICS)
        )

    return selected


def build_session_profile_url(api_url: str, session_id: str) -> str:
    parsed = urlparse(api_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"Invalid API URL: {api_url}")
    return f"{parsed.scheme}://{parsed.netloc}/session/{session_id}/profile"


def load_persona_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fp:
        payload = json.load(fp)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Persona file must contain a JSON object: {path}")

    profile = {key: payload[key] for key in PERSONA_KEYS if key in payload}
    if not profile:
        raise RuntimeError(
            f"Persona file {path} does not contain any supported keys: "
            + ", ".join(PERSONA_KEYS)
        )

    if "preferred_topics" in profile and not isinstance(
        profile["preferred_topics"], list
    ):
        raise RuntimeError("preferred_topics must be a list of strings.")
    if "avoid_topics" in profile and not isinstance(profile["avoid_topics"], list):
        raise RuntimeError("avoid_topics must be a list of strings.")
    if "explicit_notes" in profile and not isinstance(profile["explicit_notes"], list):
        raise RuntimeError("explicit_notes must be a list of strings.")

    return profile


def load_golden_set(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as fp:
        rows = json.load(fp)

    usable_rows = [row for row in rows if row.get("query")]
    if limit > 0:
        usable_rows = usable_rows[:limit]
    if not usable_rows:
        raise RuntimeError(f"No usable queries found in {path}")
    return usable_rows


def load_cached_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, list):
        return {"samples": payload}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid cache format in {path}")
    return payload


def load_cached_samples(path: Path) -> list[dict[str, Any]]:
    payload = load_cached_payload(path)
    samples = payload.get("samples", [])

    if not isinstance(samples, list):
        raise RuntimeError(f"Invalid cache format in {path}")

    return [sample for sample in samples if isinstance(sample, dict)]


def save_cached_samples(
    path: Path,
    samples: list[dict[str, Any]],
    golden_set_path: Path | None = None,
    api_url: str | None = None,
    persona_profile: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_count": len(samples),
        "golden_set_path": str(golden_set_path) if golden_set_path else None,
        "api_url": api_url,
        "persona_profile": persona_profile,
        "samples": samples,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def call_query_api(
    client: httpx.Client,
    api_url: str,
    query: str,
    session_id: str,
    use_history: bool = False,
) -> dict[str, Any]:
    response = client.post(
        api_url,
        json={
            "session_id": session_id,
            "query": query,
            "use_history": use_history,
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "answer" not in payload:
        raise RuntimeError("Unexpected /query response payload")
    return payload


def seed_persona_profile(
    client: httpx.Client,
    api_url: str,
    session_id: str,
    persona_profile: dict[str, Any],
) -> None:
    response = client.post(
        build_session_profile_url(api_url, session_id),
        json=persona_profile,
    )
    response.raise_for_status()


def build_samples(
    rows: list[dict[str, Any]],
    api_url: str,
    timeout: float,
    cache_path: Path | None = None,
    golden_set_path: Path | None = None,
    persona_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    with httpx.Client(timeout=timeout) as client:
        for index, row in enumerate(rows, start=1):
            session_id = f"ragas-vertex-{index}"
            if persona_profile:
                seed_persona_profile(
                    client=client,
                    api_url=api_url,
                    session_id=session_id,
                    persona_profile=persona_profile,
                )
            payload = call_query_api(
                client=client,
                api_url=api_url,
                query=row["query"],
                session_id=session_id,
            )
            contexts = [
                source.get("content", "")
                for source in payload.get("sources", [])
                if source.get("content")
            ]
            sample = {
                "user_input": row["query"],
                "response": payload.get("answer", ""),
                "retrieved_contexts": contexts,
            }
            if row.get("reference"):
                sample["reference"] = row["reference"]
            elif row.get("summary"):
                sample["reference"] = row["summary"]
            if row.get("reference_contexts"):
                sample["reference_contexts"] = row["reference_contexts"]
            samples.append(sample)
            if cache_path is not None:
                save_cached_samples(
                    path=cache_path,
                    samples=samples,
                    golden_set_path=golden_set_path,
                    api_url=api_url,
                    persona_profile=persona_profile,
                )
            print(
                f"[{index}/{len(rows)}] collected answer "
                f"(contexts={len(contexts)}) for query: {row['query'][:50]}"
            )

    return samples


def make_dataset(samples: list[dict[str, Any]]) -> Any:
    if EvaluationDataset is not None:
        return EvaluationDataset.from_list(samples)

    from datasets import Dataset

    legacy_dict = {
        "question": [sample["user_input"] for sample in samples],
        "answer": [sample["response"] for sample in samples],
        "contexts": [sample["retrieved_contexts"] for sample in samples],
        "ground_truth": [sample.get("reference", "") for sample in samples],
    }
    return Dataset.from_dict(legacy_dict)


def make_metrics(llm: Any, embeddings: Any, selected_metrics: list[str]) -> list[Any]:
    metric_builders: dict[str, Any] = {
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_correctness": lambda: AnswerCorrectness(
            llm=llm, embeddings=embeddings
        ),
        "answer_relevancy": lambda: AnswerRelevancy(
            llm=llm, embeddings=embeddings, strictness=1
        ),
    }
    return [metric_builders[name]() for name in selected_metrics]


def save_outputs(result: Any, csv_path: Path, json_path: Path) -> dict[str, float]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()
        dataframe.to_csv(csv_path, index=False)
        numeric_columns = [
            column
            for column in dataframe.columns
            if str(dataframe[column].dtype).startswith(("float", "int"))
        ]
        summary = {
            column: float(dataframe[column].dropna().mean())
            for column in numeric_columns
            if not dataframe[column].dropna().empty
        }
    else:
        scores = getattr(result, "scores", [])
        summary = {}
        if scores:
            metric_names = scores[0].keys()
            for name in metric_names:
                numeric_values = [
                    float(score[name])
                    for score in scores
                    if isinstance(score.get(name), (int, float))
                ]
                if numeric_values:
                    summary[name] = sum(numeric_values) / len(numeric_values)
        csv_path.write_text("", encoding="utf-8")

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()

    api_url = args.api_url
    golden_set_path = Path(args.golden_set)
    if not golden_set_path.is_absolute():
        golden_set_path = ROOT / golden_set_path
    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = ROOT / output_prefix
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_name(f"{output_prefix.name}_summary").with_suffix(
        ".json"
    )
    cache_path = Path(args.cache_path) if args.cache_path else output_prefix.with_name(
        f"{output_prefix.name}_samples"
    ).with_suffix(".json")
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    model_name = env("RAGAS_VERTEX_MODEL", "gemini-2.5-flash")
    embedding_model = env(
        "RAGAS_VERTEX_EMBEDDING_MODEL", "gemini-embedding-001"
    )
    selected_metrics = parse_metric_names(args.metrics)
    persona_profile: dict[str, Any] | None = None
    if args.persona_file:
        persona_path = Path(args.persona_file)
        if not persona_path.is_absolute():
            persona_path = ROOT / persona_path
        persona_profile = load_persona_profile(persona_path)
    run_config = RunConfig(
        timeout=args.ragas_timeout,
        max_retries=args.ragas_max_retries,
        max_wait=args.ragas_max_wait,
        max_workers=args.max_workers,
        log_tenacity=True,
    )

    google_client, use_vertex, project_id, location = build_google_client()
    rows = load_golden_set(golden_set_path, args.limit)

    if args.resume:
        if not cache_path.exists():
            raise RuntimeError(
                f"--resume was set but cache file does not exist: {cache_path}"
            )
        cached_payload = load_cached_payload(cache_path)
        cached_persona = cached_payload.get("persona_profile")
        if persona_profile != cached_persona:
            raise RuntimeError(
                "Resume cache persona does not match the requested persona. "
                "Use a fresh --cache-path/--output-prefix for this persona run."
            )
        samples = [
            sample
            for sample in cached_payload.get("samples", [])
            if isinstance(sample, dict)
        ]
        if args.limit > 0:
            samples = samples[: args.limit]
        if not samples:
            raise RuntimeError(f"Cache file is empty: {cache_path}")
        print(f"Loaded {len(samples)} cached samples from {cache_path}")
    else:
        samples = build_samples(
            rows=rows,
            api_url=api_url,
            timeout=args.timeout,
            cache_path=cache_path,
            golden_set_path=golden_set_path,
            persona_profile=persona_profile,
        )
        print(f"Saved {len(samples)} collected samples to {cache_path}")

    if not samples:
        raise RuntimeError("No evaluation samples were collected from the query API.")

    dataset = make_dataset(samples)
    llm = llm_factory(model_name, provider="google", client=google_client)
    embedding_kwargs: dict[str, Any] = {
        "client": google_client,
        "model": embedding_model,
        # The google-genai client is already configured for Vertex AI.
        # Keep use_vertex=False here so ragas uses the genai embed_content path
        # instead of the google-cloud-aiplatform TextEmbeddingModel path.
        "use_vertex": False,
    }
    if project_id:
        embedding_kwargs["project_id"] = project_id
    if location:
        embedding_kwargs["location"] = location

    embeddings = VertexGenAIEmbeddingsAdapter(**embedding_kwargs)
    metrics = make_metrics(
        llm=llm, embeddings=embeddings, selected_metrics=selected_metrics
    )

    print(f"Running ragas evaluation for {len(samples)} samples...")
    print(f"Metrics: {', '.join(selected_metrics)}")
    if persona_profile:
        print(
            "Persona: "
            + json.dumps(persona_profile, ensure_ascii=False)
        )
    print(
        "RunConfig: "
        f"max_workers={args.max_workers}, "
        f"batch_size={args.batch_size}, "
        f"timeout={args.ragas_timeout}, "
        f"max_retries={args.ragas_max_retries}, "
        f"max_wait={args.ragas_max_wait}"
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        batch_size=args.batch_size,
    )
    summary = save_outputs(result=result, csv_path=csv_path, json_path=json_path)

    print("Evaluation finished.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    if summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ragas Vertex AI evaluation failed: {exc}", file=sys.stderr)
        raise
