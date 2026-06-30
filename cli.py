"""
Entry point for all pipeline commands.

Usage:
    python cli.py ingest --subset 10
    python cli.py run [--run-name sonnet_k5] [--set llm_model=claude-sonnet-4-6]
    python cli.py eval [--run-name sonnet_k5]
    python cli.py compare baseline_haiku_k5 sonnet_k5 haiku_k10
    python cli.py --help
"""

import click
import yaml
from pathlib import Path

# Load .env if present (OPENAI_API_KEY, ANTHROPIC_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _apply_set_overrides(cfg: dict, overrides: tuple[str, ...]) -> dict:
    """Apply --set key=value overrides to config dict."""
    cfg = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise click.BadParameter(f"--set expects key=value, got: {item!r}")
        key, val = item.split("=", 1)
        # Auto-cast numeric values
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        cfg[key] = val
    return cfg


@click.group()
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml")
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.option("--subset", default=None, type=int, help="Number of contracts to pull (default: config value)")
@click.pass_context
def ingest(ctx, subset):
    """Pull CUAD data, parse contracts, write chunks to data/processed/."""
    cfg = ctx.obj["config"]
    n = subset or cfg.get("subset_size")
    click.echo(f"Ingesting {n or 'all'} contracts...")
    from ingest.pipeline import run_ingest
    run_ingest(cfg, subset=n)


@cli.command()
@click.option("--contract-id", default=None, help="Run only this contract (filename stem)")
@click.option("--force-reindex", is_flag=True, default=False, help="Re-embed all chunks even if already indexed")
@click.option("--run-name", default=None, help="Name for this run (controls output filenames)")
@click.option("--set", "overrides", multiple=True, help="Config override key=value (repeatable)")
@click.pass_context
def run(ctx, contract_id, force_reindex, run_name, overrides):
    """Embed chunks, build vector store, run agent over all (contract, clause) pairs."""
    cfg = _apply_set_overrides(ctx.obj["config"], overrides)
    # Step 3: build retrieval index
    click.echo("Building retrieval index...")
    from retrieval.retriever import Retriever
    retriever = Retriever(cfg)
    retriever.build_index(force=force_reindex)
    # Step 4: run agent
    click.echo(f"Running agent (run_name={run_name or 'default'})...")
    from agent.pipeline import run_agent
    run_agent(cfg, contract_id=contract_id, run_name=run_name)


@cli.command()
@click.option("--report-name", default=None, help="Filename stem for output report (default: run-name or 'report')")
@click.option("--run-name", default=None, help="Name of the run to score (matches agent_outputs_{name}.jsonl)")
@click.option("--skip-judge", is_flag=True, default=False, help="Skip LLM-as-judge (faster, no API cost)")
@click.option("--set", "overrides", multiple=True, help="Config override key=value (repeatable)")
@click.pass_context
def eval(ctx, report_name, run_name, skip_judge, overrides):
    """Score agent outputs against CUAD gold labels and write report."""
    cfg = _apply_set_overrides(ctx.obj["config"], overrides)
    if report_name is None:
        report_name = run_name or "report"
    click.echo(f"Scoring results (run_name={run_name or 'default'})...")
    from eval.harness import run_eval
    run_eval(cfg, report_name=report_name, run_name=run_name, skip_judge=skip_judge)


@cli.command()
@click.option("--report-name", default="taxonomy", show_default=True, help="Filename stem for output report")
@click.pass_context
def taxonomy(ctx, report_name):
    """Re-run failure taxonomy from existing scored_results.jsonl (no LLM calls)."""
    cfg = ctx.obj["config"]
    from eval.harness import run_taxonomy
    run_taxonomy(cfg, report_name=report_name)


@cli.command()
@click.argument("run_names", nargs=-1, required=True)
@click.pass_context
def compare(ctx, run_names):
    """Generate comparison report across named runs.

    Each RUN_NAME corresponds to scored_results_{name}.jsonl in data/processed/
    and uses the config recorded in that file's companion report.

    Example: python cli.py compare baseline_haiku_k5 sonnet_k5 haiku_k10
    """
    if len(run_names) < 2:
        raise click.UsageError("Need at least 2 run names to compare.")
    cfg = ctx.obj["config"]
    from eval.compare import generate_comparison
    generate_comparison(cfg, list(run_names))


if __name__ == "__main__":
    cli()
