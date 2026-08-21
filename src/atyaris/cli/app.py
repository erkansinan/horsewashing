"""Komut satiri arayuzu: gunluk bulteni ceker, kullanicinin yaris secmesini
saglar ve siralanmis, gerekceli tahminleri yazdirir.
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from atyaris.config import get_settings
from atyaris.models.entities import Race, RacePrediction
from atyaris.prediction.engine import PredictionEngine
from atyaris.services import InvalidSourceError, build_data_source, fetch_races, parse_date
from atyaris.utils.logging_config import configure_logging

app = typer.Typer(add_completion=False, help="Turkiye At Yarisi Tahmin Araci")
console = Console()


def _build_data_source(source: str, settings):
    try:
        return build_data_source(source, settings)
    except InvalidSourceError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _parse_date(target_date: str | None):
    return parse_date(target_date)


def _fetch_races(data_source, target_date, city: str | None, near: str | None) -> list[Race]:
    return fetch_races(data_source, target_date, city, near)


def _print_races(races: list[Race]) -> None:
    table = Table(title="Gunun Yarislari")
    table.add_column("#")
    table.add_column("Hipodrom")
    table.add_column("Kosu")
    table.add_column("Saat")
    table.add_column("Mesafe")
    table.add_column("Pist")
    table.add_column("Grup/Ikramiye")
    table.add_column("At Sayisi")
    for i, race in enumerate(races, start=1):
        group_prize = " / ".join(x for x in (race.group_info, race.prize_info) if x) or "-"
        table.add_row(
            str(i),
            race.hippodrome,
            str(race.race_no),
            race.start_time.strftime("%H:%M"),
            f"{race.distance_m}m",
            race.surface.value,
            group_prize,
            str(len(race.active_entries)),
        )
    console.print(table)
    for i, race in enumerate(races, start=1):
        horse_list = ", ".join(f"{e.number}-{e.horse_name}" for e in race.active_entries)
        console.print(f"  [dim]{i}. {race.hippodrome} {race.race_no}. Kosu atlari:[/dim] {horse_list}")


def _print_prediction(prediction: RacePrediction) -> None:
    race = prediction.race
    console.rule(
        f"{race.hippodrome} - {race.race_no}. Kosu ({race.distance_m}m, {race.surface.value})"
    )
    table = Table()
    table.add_column("Sira")
    table.add_column("No")
    table.add_column("At")
    table.add_column("Jokey")
    table.add_column("Skor")
    table.add_column("Etiket")
    for rank, hp in enumerate(prediction.ranked, start=1):
        table.add_row(
            str(rank),
            str(hp.entry.number),
            hp.entry.horse_name,
            hp.entry.jockey.name,
            f"{hp.score.total_score}/100",
            hp.tag,
        )
    console.print(table)
    for rank, hp in enumerate(prediction.ranked, start=1):
        console.print(
            f"\n[bold]{rank}. {hp.entry.horse_name}[/bold] ({hp.tag}, skor {hp.score.total_score}/100)"
        )
        for bullet in hp.reasoning:
            console.print(f"  - {bullet}")
    console.print(f"\n[italic yellow]{prediction.disclaimer}[/italic yellow]")


@app.command()
def bulletin(
    target_date: str = typer.Option(None, "--date", help="GG.AA.YYYY formatinda tarih (varsayilan: bugun)"),
    city: str = typer.Option(None, "--city", help="Hipodrom adina gore filtre"),
    near: str = typer.Option(None, "--near", help="Bu hipodromu listenin basina oncelikli olarak getirir"),
    source: str = typer.Option("sample", "--source", help="'sample' veya 'tjk'"),
) -> None:
    """O gunku yaris bultenini listeler."""
    configure_logging()
    settings = get_settings()
    parsed_date = _parse_date(target_date)
    data_source = _build_data_source(source, settings)
    races = _fetch_races(data_source, parsed_date, city, near)
    if not races:
        console.print("[red]Bu tarih/sehir icin yaris bulunamadi.[/red]")
        raise typer.Exit(code=1)
    _print_races(races)


@app.command()
def predict(
    races_arg: str = typer.Option(..., "--races", help="Bulten listesindeki sira no'lari, virgulle (orn. 1,3)"),
    target_date: str = typer.Option(None, "--date", help="GG.AA.YYYY formatinda tarih (varsayilan: bugun)"),
    city: str = typer.Option(None, "--city", help="Hipodrom adina gore filtre"),
    near: str = typer.Option(None, "--near", help="Bu hipodromu listenin basina oncelikli olarak getirir"),
    source: str = typer.Option("sample", "--source", help="'sample' veya 'tjk'"),
) -> None:
    """Secilen yarislar icin tahmin uretir."""
    configure_logging()
    settings = get_settings()
    parsed_date = _parse_date(target_date)
    data_source = _build_data_source(source, settings)
    races = _fetch_races(data_source, parsed_date, city, near)
    if not races:
        console.print("[red]Bu tarih/sehir icin yaris bulunamadi.[/red]")
        raise typer.Exit(code=1)
    _print_races(races)

    selected_indices = [int(x.strip()) for x in races_arg.split(",") if x.strip()]
    engine = PredictionEngine(data_source, settings)
    for idx in selected_indices:
        if idx < 1 or idx > len(races):
            console.print(f"[red]Gecersiz secim: {idx}[/red]")
            continue
        prediction = engine.predict(races[idx - 1])
        _print_prediction(prediction)


@app.command()
def interactive(
    target_date: str = typer.Option(None, "--date", help="GG.AA.YYYY formatinda tarih (varsayilan: bugun)"),
    city: str = typer.Option(None, "--city", help="Hipodrom adina gore filtre"),
    near: str = typer.Option(None, "--near", help="Bu hipodromu listenin basina oncelikli olarak getirir"),
    source: str = typer.Option("sample", "--source", help="'sample' veya 'tjk'"),
) -> None:
    """Bulteni gosterir ve yaris secimini konsoldan interaktif olarak alir."""
    configure_logging()
    settings = get_settings()
    parsed_date = _parse_date(target_date)
    data_source = _build_data_source(source, settings)
    races = _fetch_races(data_source, parsed_date, city, near)
    if not races:
        console.print("[red]Bu tarih/sehir icin yaris bulunamadi.[/red]")
        raise typer.Exit(code=1)
    _print_races(races)

    raw = typer.prompt("Tahmin uretmek istediginiz kosu numaralarini virgulle girin (orn. 1,3)")
    selected_indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
    engine = PredictionEngine(data_source, settings)
    for idx in selected_indices:
        if idx < 1 or idx > len(races):
            console.print(f"[red]Gecersiz secim: {idx}[/red]")
            continue
        prediction = engine.predict(races[idx - 1])
        _print_prediction(prediction)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Sunucunun dinleyecegi adres"),
    port: int = typer.Option(8000, "--port", help="Sunucunun dinleyecegi port"),
    reload: bool = typer.Option(False, "--reload", help="Kod degisince otomatik yeniden baslat (gelistirme icin)"),
) -> None:
    """Tarayicidan kullanilabilen basit HTML arayuzunu (FastAPI) baslatir."""
    import uvicorn

    uvicorn.run("atyaris.web.app:create_app", host=host, port=port, reload=reload, factory=True)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
