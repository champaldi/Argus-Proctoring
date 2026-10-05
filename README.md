# Proctoring — Qostanai hackathon

Desktop-приложение для прохождения теста с локальным контролем нарушений.
Интерфейс написан на PySide6, кадры с камеры передаются независимым модулям
детекции, а события сохраняются в SQLite вместе со снимками.

## Быстрый старт

Требуется Windows и Python 3.11.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

При первом запуске приложение создаст `data/proctoring.db` и каталог
`data/screenshots`. Эти файлы не попадают в Git.

## Подключение модулей команды

По умолчанию интеграция ищет:

- `phone_detector.py` с функцией `detect(frame)`;
- `gaze_analyzer.py` с функцией `analyze(frame)`;
- `environment_protection.py` с функциями `enable()` и `disable()`.

Названия модулей можно переопределить переменными окружения
`PROCTOR_PHONE_MODULE`, `PROCTOR_GAZE_MODULE` и `PROCTOR_SECURITY_MODULE`.
Пока модуля нет, приложение продолжает работать и показывает его статус в UI.

Единственный обязательный контракт — события из [events.py](events.py):

```python
from events import EventType, ProctorEvent

def detect(frame):
    return [
        ProctorEvent.create(
            EventType.PHONE_DETECTED,
            source="yolo",
            confidence=0.91,
            details={"bbox": [100, 80, 220, 310]},
        )
    ]
```

Для удобства интеграция также принимает строки и словари:

```python
return ["phone_detected"]
return [{"type": "gaze_side", "confidence": 0.82}]
```

Поддерживаемые типы перечислены в `EventType`. Интерфейс подавляет частые
дубликаты одного типа перед записью в базу.

## Настройки

Основные параметры задаются переменными окружения:

| Переменная | По умолчанию | Назначение |
| --- | ---: | --- |
| `PROCTOR_CAMERA_INDEX` | `0` | Индекс камеры OpenCV |
| `PROCTOR_ANALYSIS_FPS` | `6` | Частота запуска AI-модулей |
| `PROCTOR_PREVIEW_FPS` | `20` | Частота обновления превью |
| `PROCTOR_DATA_DIR` | `data` | Каталог базы и снимков |

## Проверка

Тесты контракта событий, риска и SQLite не требуют камеры:

```powershell
python -m unittest discover -s tests -v
```
