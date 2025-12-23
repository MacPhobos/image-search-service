"""CLI utilities using Typer."""

import asyncio

import typer
from httpx import AsyncClient

from image_search_service.scripts.faces import faces_app

app = typer.Typer(help="Image Search Service CLI")

# Register face commands subgroup
app.add_typer(faces_app)


@app.command()
def health_check() -> None:
    """Check service health by calling the health endpoint."""

    async def _check() -> None:
        async with AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8000/health", timeout=5.0)
                response.raise_for_status()
                data = response.json()
                typer.echo(f"Health check: {data}")
                if data.get("status") == "ok":
                    typer.secho("Service is healthy", fg=typer.colors.GREEN)
                else:
                    typer.secho("Service returned unexpected status", fg=typer.colors.YELLOW)
            except Exception as e:
                typer.secho(f"Health check failed: {e}", fg=typer.colors.RED)
                raise typer.Exit(1)

    asyncio.run(_check())


if __name__ == "__main__":
    app()
