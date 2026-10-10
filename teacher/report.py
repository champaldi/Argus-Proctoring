"""Отчёт по одной сессии одним HTML-файлом.

Файл самодостаточный: кадры встроены внутрь, стили и шрифты не подгружаются
из сети. Его можно открыть в любом браузере, распечатать или сохранить в PDF
и приложить как подтверждение к решению преподавателя.
"""

from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from pathlib import Path

from core.event_presentation import EVENT_LABELS, application_details
from core.storage import StoredEvent

from ui.theme import APP_NAME

from .conclusion import build_conclusion
from .data import (
    EXAM_ALLOWANCES,
    Review,
    Session,
    VERDICT_LABELS,
    allowed_event_ids,
    current_verdict,
    event_duration,
    excluded_event_ids,
    recalculate_risk,
    resolve_evidence_path,
    risk_zone,
    summarize_events,
)


ZONE_TITLES = {"low": "Зелёная зона", "medium": "Жёлтая зона", "high": "Красная зона"}
KEY_MOMENTS = 5

STYLE = """
:root { --ink:#1F2937; --text:#374151; --muted:#6B7280; --line:#E5E7EB; --paper:#FFFFFF;
        --wash:#F8F9FA; --brand:#1E3A8A;
        --low:#047857; --medium:#B45309; --high:#B91C1C;
        --low-bg:#D1FAE5; --medium-bg:#FEF3C7; --high-bg:#FEE2E2; }
* { box-sizing:border-box; }
body { margin:0; background:var(--wash); color:var(--text);
       font:15px/1.5 Inter, "Segoe UI", system-ui, -apple-system, Arial, sans-serif; }
h1, h2, b, strong, th { color:var(--ink); }
main { max-width:960px; margin:0 auto; padding:32px 20px 48px; }
h1 { font-size:26px; line-height:1.2; margin:0 0 4px; }
h2 { font-size:16px; margin:0 0 12px; }
p { margin:0 0 8px; }
.eyebrow { color:var(--brand); font-weight:700; font-size:12px; letter-spacing:.08em; text-transform:uppercase; }
.muted { color:var(--muted); }
section { background:var(--paper); border:1px solid var(--line); border-radius:12px;
          padding:20px 22px; margin-top:16px; }
.head { display:flex; gap:22px; align-items:flex-start; }
.head .who { flex:1; min-width:0; }
.photo { width:200px; flex:none; }
.photo img { width:100%; border-radius:10px; border:1px solid var(--line); display:block; }
.photo figcaption, figure figcaption { font-size:12px; color:var(--muted); margin-top:6px; }
figure { margin:0; }
.facts { display:flex; flex-wrap:wrap; gap:8px 26px; margin-top:12px; }
.facts div span { display:block; font-size:12px; color:var(--muted); }
.zone { display:inline-block; padding:4px 12px; border-radius:999px; font-weight:600; }
.zone.low { color:var(--low); background:var(--low-bg); }
.zone.medium { color:var(--medium); background:var(--medium-bg); }
.zone.high { color:var(--high); background:var(--high-bg); }
.risk { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
.risk strong { font-size:34px; line-height:1; font-variant-numeric:tabular-nums; }
.conclusion ul { margin:6px 0 10px; padding-left:20px; }
.note { font-size:13px; color:var(--muted); border-top:1px solid var(--line);
        margin-top:12px; padding-top:10px; }
table { width:100%; border-collapse:collapse; font-size:14px; }
th { text-align:left; font-weight:600; color:var(--muted); font-size:12px;
     border-bottom:1px solid var(--line); padding:6px 10px 6px 0; }
td { border-bottom:1px solid var(--line); padding:8px 10px 8px 0; vertical-align:top; }
tr:last-child td { border-bottom:0; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
tr.dismissed td { color:var(--muted); text-decoration:line-through; }
.scroll { overflow-x:auto; }
.moments { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:14px; }
.moments img { width:100%; border-radius:8px; border:1px solid var(--line); display:block; }
.moments .noframe { height:150px; border-radius:8px; border:1px dashed var(--line);
                    display:flex; align-items:center; justify-content:center; color:var(--muted); }
.moments b { display:block; margin-top:8px; }
svg.timeline { width:100%; height:44px; display:block; }
.ticks { display:flex; justify-content:space-between; font-size:12px; color:var(--muted);
         margin:0 0 10px; font-variant-numeric:tabular-nums; }
.legend { display:flex; flex-wrap:wrap; gap:6px 18px; font-size:12px; color:var(--muted); }
.legend i { display:inline-block; width:10px; height:10px; border-radius:50%;
            margin-right:6px; vertical-align:-1px; }
.verdict { font-size:18px; font-weight:600; }
footer { margin-top:18px; font-size:12px; color:var(--muted); }
@media (max-width:640px) { .head { flex-direction:column; } .photo { width:100%; max-width:280px; } }
@media print { body { background:#fff; } main { padding:0; }
               section { break-inside:avoid; border-color:#bbb; } }
"""


def _duration_text(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _image_data_uri(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    if not payload:
        return None
    kind = "png" if path.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{kind};base64,{base64.b64encode(payload).decode('ascii')}"


def _event_title(stored: StoredEvent) -> str:
    name = stored.event.type.value
    return "; ".join([EVENT_LABELS.get(name, name), *application_details(stored.event)])


def _offset_text(session: Session, stored: StoredEvent) -> str:
    seconds = max(0.0, (stored.event.occurred_at - session.started_at).total_seconds())
    return _duration_text(seconds)


def _weight_class(weight: float) -> str:
    if weight >= 30:
        return "high"
    return "medium" if weight >= 15 else "low"


def _timeline_svg(session: Session, excluded: set[str]) -> str:
    total = max(session.duration_seconds, 1.0)
    colors = {"low": "#10B981", "medium": "#F59E0B", "high": "#EF4444"}
    marks = []
    for stored in session.events:
        offset = (stored.event.occurred_at - session.started_at).total_seconds()
        x = 20 + 920 * min(1.0, max(0.0, offset / total))
        dismissed = stored.event.event_id in excluded
        color = "#9CA3AF" if dismissed else colors[_weight_class(stored.weight)]
        title = escape(f"{_offset_text(session, stored)} · {_event_title(stored)}")
        marks.append(
            f'<g><title>{title}</title>'
            f'<line x1="{x:.1f}" y1="14" x2="{x:.1f}" y2="36" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="{x:.1f}" cy="12" r="5" fill="{color}"/></g>'
        )
    return (
        '<svg class="timeline" viewBox="0 0 960 44" preserveAspectRatio="none" role="img" '
        'aria-label="Таймлайн событий сессии">'
        '<line x1="20" y1="36" x2="940" y2="36" stroke="#D1D5DB" stroke-width="2"/>'
        + "".join(marks)
        + "</svg>"
        # Подписи вынесены в HTML: растянутый по ширине SVG исказил бы текст.
        f'<div class="ticks"><span>00:00:00</span><span>{_duration_text(total)}</span></div>'
    )


def render_session_report(
    session: Session,
    review: Review | None,
    database_path: Path,
    *,
    generated_at: datetime | None = None,
) -> str:
    """Возвращает HTML отчёта; события, снятые преподавателем, в расчёт не входят."""
    review = review or Review()
    allowed = allowed_event_ids(session, review)
    excluded = set(excluded_event_ids(session, review))
    score = recalculate_risk(session, excluded)
    zone = risk_zone(score)
    conclusion = build_conclusion(session, excluded)
    verdict = current_verdict(session, review)
    generated = (generated_at or datetime.now().astimezone()).strftime("%d.%m.%Y %H:%M")
    started = session.started_at.astimezone().strftime("%d.%m.%Y %H:%M")
    status = {"active": "не завершена", "completed": "завершена"}.get(
        session.status, session.status
    )

    photo = _image_data_uri(resolve_evidence_path(database_path, session.reference_photo))
    photo_html = (
        f'<figure class="photo"><img src="{photo}" alt="Контрольный кадр студента">'
        "<figcaption>Контрольный кадр в начале теста. Личность сверяет преподаватель."
        "</figcaption></figure>"
        if photo
        else ""
    )

    score_fact = ""
    test_score = session.metadata.get("test_score")
    test_total = session.metadata.get("test_total")
    if isinstance(test_score, (int, float)) and isinstance(test_total, (int, float)):
        score_fact = (
            f"<div><span>Результат теста</span>{int(test_score)} из {int(test_total)}</div>"
        )

    reasons = "".join(f"<li>{escape(reason)}</li>" for reason in conclusion.reasons)
    reasons_html = f"<ul>{reasons}</ul>" if reasons else ""
    findings = "".join(f"<li>{escape(finding)}</li>" for finding in conclusion.findings)
    findings_html = (
        f"<p><b>Сработавшие правила</b></p><ul>{findings}</ul><p><b>События</b></p>"
        if findings
        else ""
    )

    summary = summarize_events(session, excluded)
    summary_rows = []
    for name, item in sorted(summary.items(), key=lambda pair: -pair[1].count):
        duration = f"{item.duration_seconds:.1f} с" if item.measured_count else "—"
        summary_rows.append(
            f"<tr><td>{escape(EVENT_LABELS.get(name, name))}</td>"
            f'<td class="num">{item.count}</td><td class="num">{duration}</td></tr>'
        )
    summary_html = (
        '<div class="scroll"><table><thead><tr><th>Событие</th><th class="num">Сколько раз</th>'
        '<th class="num">Длительность</th></tr></thead><tbody>'
        + "".join(summary_rows)
        + "</tbody></table></div>"
        if summary_rows
        else '<p class="muted">Подтверждённых событий нет.</p>'
    )

    moments = sorted(
        (item for item in session.events if item.event.event_id not in excluded),
        key=lambda item: (item.weight, item.event.occurred_at),
        reverse=True,
    )[:KEY_MOMENTS]
    moment_cards = []
    for stored in moments:
        image = _image_data_uri(resolve_evidence_path(database_path, stored.screenshot_path))
        frame = (
            f'<img src="{image}" alt="Кадр события">'
            if image
            else '<div class="noframe">Нет кадра</div>'
        )
        moment_cards.append(
            f"<figure>{frame}<b>{escape(_event_title(stored))}</b>"
            f"<figcaption>{_offset_text(session, stored)} от начала · "
            f"{stored.event.occurred_at.astimezone():%H:%M:%S} · вклад +{stored.weight:g}"
            "</figcaption></figure>"
        )
    moments_html = (
        f'<div class="moments">{"".join(moment_cards)}</div>'
        if moment_cards
        else '<p class="muted">Подтверждённых моментов нет.</p>'
    )

    event_rows = []
    for stored in session.events:
        dismissed = stored.event.event_id in excluded
        mark = (" (разрешено условиями экзамена)" if stored.event.event_id in allowed
                else " (снято преподавателем)")
        duration = event_duration(stored)
        event_rows.append(
            f'<tr class="{"dismissed" if dismissed else ""}">'
            f'<td class="num">{_offset_text(session, stored)}</td>'
            f"<td>{escape(_event_title(stored))}"
            f'{mark if dismissed else ""}</td>'
            f'<td class="num">+{stored.weight:g}</td>'
            f'<td class="num">{f"{duration:.1f} с" if duration is not None else "—"}</td>'
            f"<td>{escape(stored.event.source)}</td></tr>"
        )
    events_html = (
        '<div class="scroll"><table><thead><tr><th class="num">Время</th><th>Событие</th>'
        '<th class="num">Вклад</th><th class="num">Длительность</th><th>Источник</th>'
        "</tr></thead><tbody>" + "".join(event_rows) + "</tbody></table></div>"
        if event_rows
        else '<p class="muted">Событий нет.</p>'
    )

    comment = (
        f"<p>{escape(review.comment)}</p>"
        if review.comment.strip()
        else '<p class="muted">Комментария нет.</p>'
    )
    allowance_labels = [EXAM_ALLOWANCES[key][0] for key in sorted(review.allowances)
                        if key in EXAM_ALLOWANCES]
    dismissed_note = "".join((
        f'<p class="muted">Снято как ошибочные: {len(review.false_positive_ids)}.</p>'
        if review.false_positive_ids else "",
        '<p class="muted">Условия экзамена: '
        + escape("; ".join(allowance_labels)) + f". Разрешённых событий: {len(allowed)}.</p>"
        if allowance_labels else "",
    ))

    return f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{APP_NAME} · отчёт прокторинга — {escape(session.student_name)}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<section>
  <div class="head">
    <div class="who">
      <div class="eyebrow">{APP_NAME} · отчёт прокторинга</div>
      <h1>{escape(session.student_name)}</h1>
      <div class="facts">
        <div><span>Начало</span>{started}</div>
        <div><span>Длительность</span>{_duration_text(session.duration_seconds)}</div>
        <div><span>Сессия</span>{escape(status)}</div>
        {score_fact}
        <div><span>Номер сессии</span>{escape(session.id[:8])}</div>
      </div>
    </div>
    {photo_html}
  </div>
</section>
<section class="conclusion">
  <h2>Заключение системы</h2>
  <div class="risk"><strong>{score:.0f}</strong><span class="muted">из 100</span>
    <span class="zone {zone}">{ZONE_TITLES[zone]}</span></div>
  <p style="margin-top:12px"><b>{escape(conclusion.headline)}</b></p>
  {findings_html}
  {reasons_html}
  <p>{escape(conclusion.recommendation)}</p>
  <p class="note">Уровень риска — максимум за сессию. Система подсвечивает подозрительные
  моменты и не выносит решения: вердикт ставит преподаватель по кадрам.</p>
</section>
<section>
  <h2>Таймлайн</h2>
  {_timeline_svg(session, excluded)}
  <div class="legend"><span><i style="background:#10B981"></i>низкий вклад</span>
    <span><i style="background:#F59E0B"></i>средний</span>
    <span><i style="background:#EF4444"></i>высокий</span>
    <span><i style="background:#9CA3AF"></i>снято преподавателем</span></div>
</section>
<section>
  <h2>Ключевые моменты</h2>
  {moments_html}
</section>
<section>
  <h2>Сводка</h2>
  {summary_html}
</section>
<section>
  <h2>Все события · {len(session.events)}</h2>
  {events_html}
</section>
<section>
  <h2>Решение преподавателя</h2>
  <p class="verdict">{escape(VERDICT_LABELS[verdict].capitalize())}</p>
  {comment}
  {dismissed_note}
</section>
<footer>Сформировано {generated} на компьютере преподавателя. Видео не сохраняется и
никуда не передаётся; в отчёт входят только кадры зафиксированных событий.</footer>
</main>
</body>
</html>
"""


def write_session_report(
    path: Path, session: Session, review: Review | None, database_path: Path
) -> None:
    path.write_text(render_session_report(session, review, database_path), encoding="utf-8")


def default_report_name(session: Session) -> str:
    """Имя файла из имени студента и даты, без недопустимых в Windows символов."""
    safe = "".join(
        char if char.isalnum() or char in " -_" else "_" for char in session.student_name
    ).strip() or session.id[:8]
    return f"Отчёт {safe} {session.started_at.astimezone():%Y-%m-%d %H-%M}.html"
