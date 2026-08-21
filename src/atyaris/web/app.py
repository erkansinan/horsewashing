"""FastAPI tabanli basit HTML arayuzu.

``atyaris web`` komutuyla baslatilir (bkz. ``atyaris/cli/app.py``). Ayni
``atyaris.services`` yardimci fonksiyonlarini ve ``PredictionEngine``'i CLI
ile paylasir; boylece iki arayuz arasinda is mantigi tekrarlanmaz.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlencode
from types import SimpleNamespace
from xml.sax.saxutils import escape

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from atyaris.config import get_settings
from atyaris.data_sources.base import DataSourceError
from atyaris.data_sources.tjk_scraper import TJKHtmlDataSource
from atyaris.prediction.engine import PredictionEngine
from atyaris.services import InvalidSourceError, build_data_source, fetch_races, parse_date
from atyaris.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

_SORTABLE_FIELDS = {
    "number": "No",
    "total": "Toplam",
    "form": "Form",
    "jockey_trainer": "Jokey",
    "distance_surface": "Pist",
    "weight": "Agirlik",
    "rest": "Dinlenme",
}

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def create_app() -> FastAPI:
    """FastAPI uygulamasini olusturur (uvicorn factory olarak kullanilir)."""
    configure_logging()
    app = FastAPI(title="Turkiye At Yarisi Tahmin Araci", docs_url="/api/docs")

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        source: str = Query("sample", pattern="^(sample|tjk)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
    ) -> HTMLResponse:
        settings = get_settings()
        error = None
        info = None
        races = []
        hippodromes: list[str] = []
        resolved_date = date_str
        try:
            parsed_date = parse_date(date_str or None)
            resolved_date = parsed_date.isoformat()
            data_source = build_data_source(source, settings)
            if isinstance(data_source, TJKHtmlDataSource):
                hippodromes = data_source.get_available_hippodromes(parsed_date)
                if city:
                    races = fetch_races(data_source, parsed_date, city, None)
                    if not races:
                        info = (
                            "Secili hipodrom icin kosu listesi alinmadi. "
                            "TJK'nin JS/AJAX yapisi nedeniyle bu tarih icin veri gelmiyor olabilir."
                        )
                else:
                    races = []
                    info = "Lutfen listeden bir hipodrom secin."
            else:
                all_races = fetch_races(data_source, parsed_date, None, None)
                hippodromes = sorted({r.hippodrome for r in all_races})
                races = [r for r in all_races if not city or r.hippodrome == city]
        except (InvalidSourceError, ValueError) as exc:
            error = f"Gecersiz istek: {exc}"
        except DataSourceError as exc:
            error = f"Veri kaynagi hatasi: {exc}"

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "races": races,
                "error": error,
                "info": info,
                "source": source,
                "date": resolved_date,
                "city": city,
                "hippodromes": hippodromes,
            },
        )

    @app.get("/predict", response_class=HTMLResponse)
    def predict(
        request: Request,
        race_id: str,
        source: str = Query("sample", pattern="^(sample|tjk)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
        sort_by: str = Query("total", pattern="^(number|total|form|jockey_trainer|distance_surface|weight|rest)$"),
        sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    ) -> HTMLResponse:
        settings = get_settings()
        error = None
        prediction = None
        display_rows = []
        sort_urls: dict[str, str] = {}

        def _entry_key(entry) -> str:  # type: ignore[no-untyped-def]
            if entry.source_horse_id is not None:
                return f"source:{entry.source_horse_id}"
            return f"horse:{entry.horse_id}"

        def _build_predict_url(*, sort_field: str, sort_direction: str) -> str:
            return "/predict?" + urlencode(
                {
                    "race_id": race_id,
                    "source": source,
                    "date": date_str,
                    "city": city,
                    "sort_by": sort_field,
                    "sort_order": sort_direction,
                }
            )

        try:
            parsed_date = parse_date(date_str or None)
            data_source = build_data_source(source, settings)
            races = fetch_races(data_source, parsed_date, city or None, None)
            race = next((r for r in races if r.id == race_id), None)
            if race is None:
                error = "Yaris bulunamadi (bulten degismis olabilir; lutfen tekrar secin)."
            else:
                engine = PredictionEngine(data_source, settings)
                prediction = engine.predict(race)
                key_map = {
                    "number": lambda hp: hp.entry.number,
                    "total": lambda hp: hp.score.total_score,
                    "form": lambda hp: hp.score.form_score,
                    "jockey_trainer": lambda hp: hp.score.jockey_trainer_score,
                    "distance_surface": lambda hp: hp.score.distance_surface_score,
                    "weight": lambda hp: hp.score.weight_score,
                    "rest": lambda hp: hp.score.rest_score,
                }
                reverse = sort_order == "desc"
                ranked_by_key = {_entry_key(hp.entry): hp for hp in prediction.ranked}
                active_rows = [
                    SimpleNamespace(
                        entry=hp.entry,
                        score=hp.score,
                        reasoning=hp.reasoning,
                        tag=hp.tag,
                        is_scratched=False,
                    )
                    for hp in prediction.ranked
                ]

                unscored_active_rows = [
                    SimpleNamespace(
                        entry=entry,
                        score=None,
                        reasoning=["Bu at aktif durumda; ancak istatistik verisi alinamadigi icin skorlanamadi."],
                        tag="Veri Eksik",
                        is_scratched=False,
                    )
                    for entry in race.entries
                    if not entry.is_scratched and _entry_key(entry) not in ranked_by_key
                ]

                scratched_rows = [
                    SimpleNamespace(
                        entry=entry,
                        score=None,
                        reasoning=["Bu at kosmaz (scratch) olarak isaretli."],
                        tag="Koşmaz",
                        is_scratched=True,
                    )
                    for entry in race.entries
                    if entry.is_scratched
                ]

                if sort_by == "number":
                    display_rows = sorted(
                        active_rows + unscored_active_rows + scratched_rows,
                        key=lambda row: row.entry.number,
                        reverse=reverse,
                    )
                else:
                    display_rows = sorted(active_rows, key=key_map[sort_by], reverse=reverse) + sorted(
                        unscored_active_rows, key=lambda row: row.entry.number
                    ) + sorted(
                        scratched_rows, key=lambda row: row.entry.number
                    )

                for field in _SORTABLE_FIELDS:
                    next_direction = "asc" if field == sort_by and sort_order == "desc" else "desc"
                    if field == sort_by and sort_order == "asc":
                        next_direction = "desc"
                    sort_urls[field] = _build_predict_url(sort_field=field, sort_direction=next_direction)
        except (InvalidSourceError, ValueError) as exc:
            error = f"Gecersiz istek: {exc}"
        except RuntimeError as exc:
            error = (
                f"Tahmin uretilemedi: {exc} "
                "(TJK kaynaginda at istatistikleri su an sinirli/erisilemez olabilir)."
            )
        except DataSourceError as exc:
            error = f"Veri kaynagi hatasi: {exc}"

        return templates.TemplateResponse(
            request,
            "predict.html",
            {
                "prediction": prediction,
                "rows": display_rows,
                "error": error,
                "source": source,
                "date": date_str,
                "city": city,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "sort_urls": sort_urls,
                "sortable_fields": _SORTABLE_FIELDS,
            },
        )

    @app.get("/predict-all-pdf")
    def predict_all_pdf(
        source: str = Query("sample", pattern="^(sample|tjk)$"),
        date_str: str = Query("", alias="date"),
        city: str = Query(""),
    ) -> Response:
        settings = get_settings()
        if not city:
            raise HTTPException(status_code=400, detail="PDF olusturmak icin once bir hipodrom secin.")

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            import reportlab
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail="PDF olusturma icin 'reportlab' kurulu olmali. Lutfen bagimliliklari guncelleyin.",
            ) from exc

        parsed_date = parse_date(date_str or None)
        data_source = build_data_source(source, settings)
        races = fetch_races(data_source, parsed_date, city or None, None)
        if not races:
            raise HTTPException(
                status_code=400,
                detail="Secili tarih/hipodrom icin yaris bulunamadi; PDF olusturulamadi.",
            )

        engine = PredictionEngine(data_source, settings)
        from io import BytesIO

        def _resolve_pdf_font_paths() -> tuple[Path | None, Path | None]:
            candidates = [
                (
                    Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf",
                    Path(reportlab.__file__).resolve().parent / "fonts" / "VeraBd.ttf",
                ),
                (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
                (Path("C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")),
                (
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                ),
                (Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")),
                Path("C:/Windows/Fonts/arial.ttf"),
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
            ]
            for candidate in candidates:
                if isinstance(candidate, tuple):
                    regular, bold = candidate
                    if regular.exists() and bold.exists():
                        return regular, bold
                elif candidate.exists():
                    return candidate, None
            return None, None

        body_font_name = "Helvetica"
        header_font_name = "Helvetica-Bold"
        body_font_path, header_font_path = _resolve_pdf_font_paths()
        if body_font_path is not None:
            body_font_name = "TurkishSans"
            if body_font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(body_font_name, str(body_font_path)))
            if header_font_path is not None:
                header_font_name = "TurkishSansBold"
                if header_font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(header_font_name, str(header_font_path)))
            else:
                header_font_name = body_font_name

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=12 * mm,
            rightMargin=12 * mm,
            topMargin=12 * mm,
            bottomMargin=12 * mm,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleTurkish",
            parent=styles["Heading2"],
            fontName=header_font_name,
            fontSize=13,
            leading=16,
        )
        info_style = ParagraphStyle(
            "InfoTurkish",
            parent=styles["Normal"],
            fontName=body_font_name,
            fontSize=9,
            leading=12,
        )
        race_style = ParagraphStyle(
            "RaceTurkish",
            parent=styles["Normal"],
            fontName=header_font_name,
            fontSize=10,
            leading=13,
        )
        cell_style = ParagraphStyle(
            "CellTurkish",
            parent=styles["Normal"],
            fontName=body_font_name,
            fontSize=8,
            leading=10,
        )
        cell_numeric_style = ParagraphStyle(
            "CellNumericTurkish",
            parent=cell_style,
            alignment=2,
        )
        header_cell_style = ParagraphStyle(
            "HeaderCellTurkish",
            parent=cell_style,
            fontName=header_font_name,
            fontSize=8.5,
            leading=10.5,
            alignment=1,
        )

        elements = [
            Paragraph(
                escape(f"Tahmin Raporu - {city} - {parsed_date.strftime('%d.%m.%Y')}"),
                title_style,
            ),
            Paragraph(escape(f"Kaynak: {source}"), info_style),
            Spacer(1, 6),
        ]

        for race in races:
            prediction = engine.predict(race)
            elements.append(
                Paragraph(
                    escape(
                        f"Koşu {race.race_no} ({race.start_time.strftime('%H:%M')}) - "
                        f"{race.distance_m}m {race.surface.value}"
                    ),
                    race_style,
                )
            )
            rows = [
                [
                    Paragraph("No", header_cell_style),
                    Paragraph("At", header_cell_style),
                    Paragraph("Toplam", header_cell_style),
                    Paragraph("Form", header_cell_style),
                    Paragraph("Jokey", header_cell_style),
                    Paragraph("Pist", header_cell_style),
                    Paragraph("Ağırlık", header_cell_style),
                    Paragraph("Dinlenme", header_cell_style),
                ]
            ]
            for hp in prediction.ranked:
                rows.append(
                    [
                        Paragraph(str(hp.entry.number), cell_numeric_style),
                        Paragraph(escape(hp.entry.horse_name), cell_style),
                        Paragraph(f"{hp.score.total_score:.1f}", cell_numeric_style),
                        Paragraph(f"{hp.score.form_score:.1f}", cell_numeric_style),
                        Paragraph(f"{hp.score.jockey_trainer_score:.1f}", cell_numeric_style),
                        Paragraph(f"{hp.score.distance_surface_score:.1f}", cell_numeric_style),
                        Paragraph(f"{hp.score.weight_score:.1f}", cell_numeric_style),
                        Paragraph(f"{hp.score.rest_score:.1f}", cell_numeric_style),
                    ]
                )

            table = LongTable(
                rows,
                colWidths=[10 * mm, 56 * mm, 16 * mm, 14 * mm, 16 * mm, 14 * mm, 18 * mm, 19 * mm],
                repeatRows=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 8))

        doc.build(elements)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        file_name = f"tahmin-raporu-{city}-{parsed_date.isoformat()}.pdf".replace(" ", "_")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
        )

    return app
