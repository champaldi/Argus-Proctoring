# Proctoring — Qostanai hackathon

Локальное Windows-приложение для прохождения теста с автоматической фиксацией
подозрительных событий. Один поток камеры передаёт одинаковый кадр модулю YOLO
для телефона и модулю MediaPipe для лица, головы и взгляда. События сохраняются
в SQLite вместе со снимками и формируют уровень риска от 0 до 100.

## Быстрый старт

Требуется 64-разрядный Python 3.11 и интернет при первом запуске для загрузки
весов YOLO.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

При первом запуске создаются `data/proctoring.db` и `data/screenshots`.
Вес `yolov8s.pt` скачивается Ultralytics автоматически и не хранится в Git.

## Архитектура

- `main.py` — единая точка запуска;
- `ui/main_window.py` — тест, превью, статусы, риск и итоговый экран;
- `core/detectors.py` — адаптеры независимых детекторов;
- `core/pipeline.py` — очередь, эпизоды и запись событий;
- `core/storage.py` — SQLite и снимки нарушений;
- `core/risk.py` — уровень риска 0–100;
- `detection/phone_detector.py` — YOLOv8, телефон и наведение на экран;
- `gaze_analyzer.py` — MediaPipe Face Mesh, голова, взгляд и число лиц;
- `events.py` — единый контракт событий команды.

Камера открывается только в `CameraWorker`. Полученный BGR-кадр последовательно
передаётся обоим анализаторам; сами модули камеру не открывают.

## Контракт событий

Детектор возвращает список объектов `ProctorEvent`, строк или словарей:

```python
from events import EventType, ProctorEvent

def detect(frame):
    return [
        ProctorEvent.create(
            EventType.PHONE_DETECTED,
            source="yolo",
            confidence=0.91,
            details={"bbox": [100, 80, 220, 310], "phone_count": 1},
        )
    ]
```

```python
return [{"type": "gaze_side", "duration": 3.2, "face_count": 1}]
```

Поддерживаемые значения находятся в `EventType`. Событие создаётся один раз за
непрерывный эпизод и снова учитывается только после прекращения и нового начала
нарушения.

## Модули

Телефонный модуль по умолчанию подключается как
`detection.phone_detector.detect(frame)`. Используется `yolov8s.pt`, а анализ
выполняется на каждом третьем кадре. Если совместная работа с MediaPipe слишком
медленная, замените `MODEL_NAME` на `"yolov8n.pt"` в
`detection/phone_detector.py`.

Модуль взгляда подключается как `gaze_analyzer.analyze(frame)`. Он использует
`mediapipe==0.10.21`, Face Mesh с радужками и OpenCV `solvePnP`. Пороговые
значения находятся в `AnalyzerConfig`. Перед финальным показом их нужно
проверить при реальном освещении, в очках и с разными положениями головы.

Модуль защиты подключается как `environment_protection` и должен предоставлять
`enable()` и `disable()`. Если функция `enable` принимает callback, приложение
передаёт в неё обработчик событий.

Имена модулей можно переопределить переменными окружения
`PROCTOR_PHONE_MODULE`, `PROCTOR_GAZE_MODULE` и `PROCTOR_SECURITY_MODULE`.

## Настройки

| Переменная | По умолчанию | Назначение |
| --- | ---: | --- |
| `PROCTOR_CAMERA_INDEX` | `0` | Индекс камеры OpenCV |
| `PROCTOR_ANALYSIS_FPS` | `6` | Частота общего вызова AI-модулей |
| `PROCTOR_PREVIEW_FPS` | `20` | Частота обновления превью |
| `PROCTOR_DATA_DIR` | `data` | Каталог базы и снимков |

Веса, границы цветовых зон, множитель сочетания событий и скорость снижения
риска находятся в `config.py`.

## Проверка

Автоматические тесты логики не требуют камеры или скачивания моделей:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s detection -p "test_*.py" -v
```

Перед демонстрацией отдельно проверьте:

- телефон под разными углами и наведение на экран;
- взгляд в сторону, вниз и обычный взгляд на клавиатуру;
- отсутствие лица и второго человека;
- плохое освещение и очки;
- запись SQLite, снимки и итоговый экран преподавателя;
- корректное освобождение камеры и выключение защиты.

## Готовые компоненты и AI

Проект использует Ultralytics YOLOv8, MediaPipe 0.10.21, OpenCV, NumPy,
PySide6, SQLite, keyboard, pywin32 и psutil. При разработке кода, тестов и
документации использовался AI-помощник Codex; это нужно указать в материалах
хакатона.
