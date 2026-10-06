# Защита окружения

Windows-модуль для общего приложения. Публичный контракт:
`enable(callback=None, *, hwnd=None)`, `disable()`, `status()` и
`drain_events()`. Импорт модуля не устанавливает хуки и не изменяет систему.

## Возможности

- блокировка Alt+Tab, Win, Ctrl+C/V/X, PrtScn и сочетаний переключения вкладок;
- блокировка сочетаний выхода из теста: Alt+F4, Alt+Esc, Alt+Space, Ctrl+Esc,
  Ctrl+Shift+Esc (Ctrl+Alt+Del и Win+L Windows перехватывать не позволяет);
- очистка буфера обмена при старте и завершении теста
  (`ProtectionConfig.clear_clipboard`);
- контроль активного окна и попытка вернуть фокус в окно теста;
- окно теста разворачивается и закрепляется поверх всех окон на время сессии
  (`ProtectionConfig.lock_window`); это не системный kiosk-режим, другие topmost-окна
  и системные экраны могут отображаться поверх;
- скрытие окна от поддерживаемого программного захвата через
  `WDA_EXCLUDEFROMCAPTURE`;
- поиск браузеров, мессенджеров и удалённого ПО среди процессов и работающих
  Windows-служб;
- определение RDP через `SM_REMOTESESSION`;
- постоянный контроль количества активных дисплеев через `SM_CMONITORS`;
- обнаружение программно внедрённых событий клавиатуры и мыши по флагам
  `LLKHF_INJECTED` и `LLMHF_INJECTED`.

Удалённое ПО: AnyDesk, RustDesk, Parsec, TeamViewer, распространённые VNC,
Chrome Remote Desktop, Quick Assist и Windows Remote Assistance. Обнаруженные
процессы не завершаются принудительно — они записываются как доказательства.

## Подключение

Общее приложение уже передаёт HWND окна PySide6:

```python
import environment_protection as protection

try:
    protection.enable(on_event, hwnd=int(test_window.winId()))
    run_test_event_loop()
finally:
    protection.disable()
```

Вызывать `enable()` нужно после показа окна. Без callback события читаются через
`drain_events()`. Текущее состояние доступно через `status()`.

## События

- `hotkey_blocked` — заблокированная комбинация;
- `window_switched` — уход фокуса и результат его восстановления;
- `suspicious_process` — новые процессы и службы в одном событии;
- `capture_protection_failed` — Windows не включила защиту захвата;
- `remote_session` — приложение запущено через RDP;
- `multiple_monitors` — активно больше одного дисплея;
- `injected_input` — программно внедрённая мышь или клавиатура.

`multiple_monitors` и `injected_input` создаются один раз на непрерывный
эпизод. Для внедрённого ввода новый эпизод начинается после секунды тишины.

## Безопасное выключение

- `Ctrl+Alt+F12` немедленно прекращает защиту;
- `disable()` повторяем и снимает защиту захвата, keyboard hook и оба
  низкоуровневых input hook;
- защита также выключается при закрытии окна, ошибке, переполнении очереди или
  по таймеру (по умолчанию 4 часа, `ProtectionConfig.max_seconds`);
- `atexit` вызывает `disable()` при обычном завершении Python.

Это демонстрационная защита, а не системный киоск. Ctrl+Alt+Del не блокируется.
`WDA_EXCLUDEFROMCAPTURE` не защищает от внешней камеры, аппаратной карты
захвата и некоторых нестандартных драйверов.

## Проверка

```powershell
python -m unittest discover -s tests -v
python -m environment_protection.demo --seconds 60
```

Автоматические тесты проверяют lifecycle, события, дедупликацию, процессы,
службы, RDP, мониторы, распознавание injected-флагов и реальный краткий
start/stop неблокирующих Windows-хуков.
