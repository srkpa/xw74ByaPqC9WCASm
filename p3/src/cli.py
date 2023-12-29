import click
from pathlib import Path
from src.manual_ranking import ranking
from src.manual_ranking.typings import EmbedderConfig
from src.learning_to_rank import train


class EmbedderConfigCommand(click.Group):
    def invoke(self, ctx):
        embedder_config: EmbedderConfig = {
            "name": ctx.params.get("embedder_name"),
            "type": ctx.params.get("embedder_type"),
        }
        ctx.params["embedder_config"] = embedder_config
        del ctx.params["embedder_name"]
        del ctx.params["embedder_type"]
        super().invoke(ctx)


@click.group(cls=EmbedderConfigCommand)
@click.option(
    "-f",
    "--filepath",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "-en",
    "--embedder-name",
    type=click.Choice(
        [
            "bert-base-uncased",
            "bert-large-uncased",
            "bert-base-cased",
            "bert-large-cased",
            "bert-large-uncased-whole-word-masking",
            "distilbert-base-uncased",
            "albert-base-v2",
            "tdidf",
            "bag-of-words",
        ]
    ),
    default="bert-base-uncased",
    required=False,
)
@click.option(
    "-et",
    "--embedder-type",
    type=click.Choice(
        [
            "sk",
            "hf",
        ]
    ),
    default="hf",
    required=False,
)
@click.option("-q", "--query", type=str, required=False, default=None)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable DEBUG option (run pipeline on a small portion of the csv file)",
)
@click.pass_context
def cli(ctx, query: str, filepath: Path, embedder_config: EmbedderConfig, debug: bool):
    ctx.ensure_object(dict)

    ctx.obj.update(
        {
            "query": query,
            "filepath": filepath,
            "embedder_config": embedder_config,
            "debug": debug,
        }
    )


@cli.command()
@click.pass_context
def rank(ctx):
    ranking.main(
        query=ctx.obj["query"],
        file_path=ctx.obj["filepath"],
        embedder_config=ctx.obj["embedder_config"],
        debug=ctx.obj["debug"],
    )


@cli.command()
@click.pass_context
def fit(ctx):
    train.main(
        file_path=ctx.obj["filepath"],
        embbeder_config=ctx.obj["embedder_config"],
        debug=ctx.obj["debug"],
    )


if __name__ == "__main__":
    cli(obj={})
