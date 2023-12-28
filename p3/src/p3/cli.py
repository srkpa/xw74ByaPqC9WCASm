import click
from pathlib import Path
from p3.ranking import main
from p3.typings import EmbedderConfig


class EmbedderConfigCommand(click.Command):
    def invoke(self, ctx):
        embedder_config: EmbedderConfig = {
            "name": ctx.params.get("embedder_name"),
            "type": ctx.params.get("embedder_type"),
        }
        ctx.params["embedder_config"] = embedder_config
        del ctx.params["embedder_name"]
        del ctx.params["embedder_type"]
        super().invoke(ctx)


@click.group()
def cli():
    ...


@cli.command(cls=EmbedderConfigCommand)
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
@click.option(
    "-q",
    "--query",
    type=str,
    required=True,
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable DEBUG option (run pipeline on a small portion of the csv file)",
)
def rank(query: str, filepath: Path, embedder_config: EmbedderConfig, debug: bool):
    main(query, filepath, embedder_config, debug)


if __name__ == "__main__":
    cli()
