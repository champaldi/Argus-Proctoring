"""Protection behavior with simulated OS boundaries; never locks the desktop."""

import importlib.util
import threading
import time
import unittest
from types import SimpleNamespace


class ModuleContractTests(unittest.TestCase):
    def test_team_module_is_available(self):
        self.assertIsNotNone(importlib.util.find_spec("environment_protection"))


class FakeDesktop:
    def __init__(self):
        self.foreground = 100
        self.alive = True
        self.restore_succeeds = True
        self.restores = []
        self.hook = None
        self.process_list = []
        self.service_list = []
        self.fail_install = False
        self.fail_remove = False
        self.modifiers = set()
        self.capture_succeeds = True
        self.capture_calls = []
        self.remote_session = False

    def resolve_target(self, hwnd):
        return (hwnd or 100), 10

    def initial_modifiers(self):
        return self.modifiers

    def protect_capture(self, hwnd):
        self.capture_calls.append(("protect", hwnd))
        return self.capture_succeeds

    def release_capture(self, hwnd):
        self.capture_calls.append(("release", hwnd))
        return True

    def is_remote_session(self):
        return self.remote_session

    def install_hook(self, callback):
        if self.fail_install:
            raise RuntimeError("hook installation failed")
        self.hook = callback
        return self.remove_hook

    def remove_hook(self):
        if self.fail_remove:
            raise RuntimeError("hook removal failed")
        self.hook = None

    def target_exists(self, hwnd, pid):
        return self.alive

    def foreground_window(self):
        return self.foreground

    def belongs_to_target(self, hwnd, target, pid):
        return hwnd in (100, 101)  # 101 is an owned test dialog.

    def restore(self, hwnd):
        self.restores.append(hwnd)
        if self.restore_succeeds:
            self.foreground = hwnd
        return self.restore_succeeds

    def processes(self):
        return list(self.process_list)

    def services(self):
        return list(self.service_list)


def key(name, scan_code, kind="down"):
    return SimpleNamespace(name=name, scan_code=scan_code, event_type=kind)


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        from environment_protection.controller import Protection, ProtectionConfig

        self.desktop = FakeDesktop()
        self.events = []
        self.protection = Protection(
            self.desktop,
            ProtectionConfig(poll_interval=0.01, process_interval=0.02, max_seconds=5),
        )
        self.addCleanup(self.protection.disable)

    def start(self):
        self.protection.enable(self.events.append)
        return self.desktop.hook

    def test_required_shortcuts_and_russian_layout_are_blocked(self):
        hook = self.start()
        for modifier, modifier_scan, name, scan, expected in [
            ("alt", 56, "tab", 15, "alt+tab"),
            ("ctrl", 29, "c", 46, "ctrl+c"),
            ("ctrl", 29, "м", 47, "ctrl+v"),
            ("ctrl", 29, "tab", 15, "ctrl+tab"),
        ]:
            self.assertTrue(hook(key(modifier, modifier_scan)))
            self.assertFalse(hook(key(name, scan)))
            self.assertFalse(hook(key(name, scan)))  # Auto-repeat is one event.
            self.assertFalse(hook(key(name, scan, "up")))
            self.assertTrue(hook(key(modifier, modifier_scan, "up")))
            self.assertTrue(
                wait_for(
                    lambda expected=expected: any(
                        e["details"].get("hotkey") == expected for e in self.events
                    )
                )
            )
        for name, scan in [("left windows", 91), ("right windows", 92), ("print screen", 55)]:
            self.assertFalse(hook(key(name, scan)))
            self.assertFalse(hook(key(name, scan, "up")))
        self.assertTrue(hook(key("a", 30)))
        self.assertTrue(hook(key("a", 30, "up")))
        self.assertTrue(wait_for(lambda: len(self.events) == 7))

    def test_modifiers_held_before_enable_are_respected(self):
        self.desktop.modifiers = {"ctrl"}
        hook = self.start()
        self.assertFalse(hook(key("c", 46)))

    def test_releasing_right_modifier_keeps_held_left_modifier(self):
        hook = self.start()
        for left, right, scan, target, target_scan in [
            ("ctrl", "right ctrl", 29, "c", 46),
            ("alt", "right alt", 56, "tab", 15),
        ]:
            hook(key(left, scan))
            hook(key(right, scan))
            hook(key(right, scan, "up"))
            self.assertFalse(hook(key(target, target_scan)))
            hook(key(target, target_scan, "up"))
            hook(key(left, scan, "up"))
            self.assertTrue(hook(key(target, target_scan)))
            hook(key(target, target_scan, "up"))

    def test_disable_is_idempotent_and_stale_hook_passes_keys(self):
        hook = self.start()
        self.protection.enable(self.events.append)
        self.assertIs(self.desktop.hook, hook)
        self.protection.disable()
        self.protection.disable()
        self.assertIsNone(self.desktop.hook)
        self.assertTrue(hook(key("left windows", 91)))
        self.assertFalse(self.protection.status()["enabled"])

    def test_capture_protection_is_applied_and_released_once(self):
        self.start()
        self.assertTrue(self.protection.status()["capture_protected"])
        self.protection.disable()
        self.protection.disable()
        self.assertEqual(
            self.desktop.capture_calls,
            [("protect", 100), ("release", 100)],
        )
        self.assertFalse(self.protection.status()["capture_protected"])

    def test_capture_protection_failure_is_reported_without_locking_desktop(self):
        self.desktop.capture_succeeds = False
        self.start()
        self.assertTrue(wait_for(lambda: bool(self.events)))
        self.assertEqual(self.events[0]["type"], "capture_protection_failed")
        self.assertEqual(self.events[0]["details"], {"target_hwnd": 100})
        self.assertFalse(self.protection.status()["capture_protected"])

    def test_remote_windows_session_is_reported_once_at_start(self):
        self.desktop.remote_session = True
        self.start()
        self.assertTrue(wait_for(lambda: bool(self.events)))
        self.assertEqual([event["type"] for event in self.events], ["remote_session"])
        self.assertEqual(self.events[0]["details"], {"protocol": "rdp"})

    def test_emergency_shortcut_releases_keyboard_even_with_stuck_callback(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def stuck(event):
            entered.set()
            release.wait(3)

        self.protection.enable(stuck)
        hook = self.desktop.hook
        hook(key("print screen", 55))
        self.assertTrue(entered.wait(1))
        hook(key("ctrl", 29))
        hook(key("alt", 56))
        self.assertTrue(hook(key("f12", 88)))
        self.assertTrue(hook(key("left windows", 91)))
        self.assertTrue(wait_for(lambda: self.desktop.hook is None))
        self.assertEqual(self.protection.status()["reason"], "emergency_hotkey")

    def test_focus_switch_reports_once_until_return_and_allows_owned_dialog(self):
        self.desktop.restore_succeeds = False
        self.start()
        self.desktop.foreground = 200
        self.assertTrue(wait_for(lambda: len(self.events) == 1))
        time.sleep(0.06)
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["type"], "window_switched")
        self.assertFalse(self.events[0]["details"]["focus_restored"])
        self.assertTrue(self.desktop.restores)
        self.desktop.foreground = 101
        time.sleep(0.04)
        self.assertEqual(len(self.events), 1)
        self.desktop.restore_succeeds = True
        self.desktop.foreground = 200
        self.assertTrue(wait_for(lambda: len(self.events) == 2))
        self.assertEqual(self.desktop.foreground, 100)

    def test_target_disappearing_releases_protection(self):
        self.start()
        self.desktop.alive = False
        self.assertTrue(wait_for(lambda: self.desktop.hook is None))
        self.assertEqual(self.protection.status()["reason"], "target_closed")

    def test_successful_restore_does_not_hide_an_immediate_new_switch(self):
        self.start()
        self.desktop.foreground = 200
        self.assertTrue(wait_for(lambda: len(self.events) == 1))
        self.desktop.foreground = 200
        self.assertTrue(wait_for(lambda: len(self.events) == 2))
        self.assertEqual(self.desktop.foreground, 100)

    def test_old_hook_cannot_block_new_session(self):
        old_hook = self.start()
        self.protection.disable()
        new_hook = self.start()
        self.assertTrue(old_hook(key("left windows", 91)))
        self.assertFalse(new_hook(key("left windows", 91)))

    def test_start_failure_after_hook_installation_rolls_back(self):
        from unittest.mock import patch

        with patch("threading.Thread.start", side_effect=RuntimeError("thread start failed")):
            with self.assertRaisesRegex(RuntimeError, "thread start"):
                self.start()
        self.assertIsNone(self.desktop.hook)
        self.assertFalse(self.protection.status()["enabled"])

    def test_monitor_failure_releases_keys(self):
        from unittest.mock import patch

        with patch.object(self.desktop, "processes", side_effect=RuntimeError("scan failed")):
            self.start()
            self.assertTrue(wait_for(lambda: self.desktop.hook is None))
        self.assertFalse(self.protection.status()["enabled"])
        self.assertIn("scan failed", self.protection.status()["last_error"])

    def test_no_callback_events_can_be_drained_and_overflow_releases_keys(self):
        self.protection.enable()
        hook = self.desktop.hook
        self.assertFalse(hook(key("print screen", 55)))
        hook(key("print screen", 55, "up"))
        events = self.protection.drain_events()
        self.assertEqual([e["type"] for e in events], ["hotkey_blocked"])
        self.assertEqual(self.protection.drain_events(), [])
        for _ in range(257):
            hook(key("print screen", 55))
            hook(key("print screen", 55, "up"))
        self.assertTrue(hook(key("left windows", 91)))
        self.assertEqual(self.protection.status()["reason"], "event_queue_full")

    def test_timeout_releases_without_user_interaction(self):
        from environment_protection.controller import Protection, ProtectionConfig

        self.protection = Protection(self.desktop, ProtectionConfig(max_seconds=0.05))
        self.start()
        self.assertTrue(wait_for(lambda: self.desktop.hook is None))
        self.assertEqual(self.protection.status()["reason"], "timeout")

    def test_partial_start_failure_leaves_disabled(self):
        self.desktop.fail_install = True
        with self.assertRaisesRegex(RuntimeError, "installation"):
            self.start()
        self.assertFalse(self.protection.status()["enabled"])
        self.assertIsNone(self.desktop.hook)

    def test_unhook_failure_does_not_leave_keys_blocked_and_is_retryable(self):
        hook = self.start()
        self.desktop.fail_remove = True
        self.protection.disable()
        self.assertTrue(hook(key("left windows", 91)))
        self.assertIn("removal", self.protection.status()["last_error"])
        self.desktop.fail_remove = False
        self.protection.disable()
        self.assertIsNone(self.desktop.hook)

    def test_callback_error_disables_protection(self):
        def broken(event):
            raise RuntimeError("event sink failed")

        self.protection.enable(broken)
        self.desktop.hook(key("print screen", 55))
        self.assertTrue(wait_for(lambda: self.desktop.hook is None))
        self.assertIn("event sink", self.protection.status()["last_error"])

    def test_normal_disable_delivers_already_queued_events(self):
        entered, release = threading.Event(), threading.Event()
        received = []
        self.addCleanup(release.set)

        def delayed(event):
            entered.set()
            release.wait(1)
            received.append(event)

        self.protection.enable(delayed)
        hook = self.desktop.hook
        hook(key("print screen", 55))
        hook(key("print screen", 55, "up"))
        self.assertTrue(entered.wait(1))
        hook(key("left windows", 91))
        stopper = threading.Thread(target=self.protection.disable)
        stopper.start()
        self.assertTrue(wait_for(lambda: self.desktop.hook is None))
        release.set()
        stopper.join(1)
        self.assertFalse(stopper.is_alive())
        self.assertEqual(len(received), 2)

    def test_processes_ignore_target_tree_and_deduplicate_browser_children(self):
        self.desktop.process_list = [
            {"pid": 10, "ppid": 1, "name": "chrome.exe", "create_time": 1},
            {"pid": 11, "ppid": 10, "name": "chrome.exe", "create_time": 1},
            {"pid": 20, "ppid": 1, "name": "msedge.exe", "create_time": 2},
            {"pid": 21, "ppid": 20, "name": "msedge.exe", "create_time": 2},
            {"pid": 30, "ppid": 1, "name": "Telegram.exe", "create_time": 3},
        ]
        self.start()
        self.assertTrue(wait_for(lambda: len(self.events) == 1))
        time.sleep(0.06)
        self.assertEqual(len(self.events), 1)
        self.assertEqual({p["pid"] for p in self.events[0]["details"]["processes"]}, {20, 30})
        self.assertTrue(all(e["type"] == "suspicious_process" for e in self.events))
        self.desktop.process_list[-1]["create_time"] = 4  # PID reuse is a new process.
        self.assertTrue(wait_for(lambda: len(self.events) == 2))

    def test_remote_access_process_families_are_reported(self):
        names = [
            "AnyDesk.exe",
            "rustdesk.exe",
            "parsecd.exe",
            "TeamViewer.exe",
            "winvnc.exe",
            "remoting_host.exe",
            "QuickAssist.exe",
        ]
        self.desktop.process_list = [
            {"pid": 20 + index, "ppid": 1, "name": name, "create_time": index}
            for index, name in enumerate(names)
        ]
        self.start()
        self.assertTrue(wait_for(lambda: bool(self.events)))
        reported = {item["name"].casefold() for item in self.events[0]["details"]["processes"]}
        self.assertEqual(reported, {name.casefold() for name in names})

    def test_remote_access_services_share_one_software_event_with_processes(self):
        self.desktop.process_list = [
            {"pid": 20, "ppid": 1, "name": "AnyDesk.exe", "create_time": 1}
        ]
        self.desktop.service_list = [
            {"name": "TeamViewer", "display_name": "TeamViewer", "status": "running"},
            {"name": "chromoting", "display_name": "Chrome Remote Desktop", "status": "running"},
            {"name": "RustDesk", "display_name": "RustDesk Service", "status": "stopped"},
        ]
        self.start()
        self.assertTrue(wait_for(lambda: bool(self.events)))
        self.assertEqual(len(self.events), 1)
        details = self.events[0]["details"]
        self.assertEqual({item["name"] for item in details["services"]}, {"TeamViewer", "chromoting"})
        self.assertEqual(details["count"], 3)

    def test_events_pass_existing_adapter_and_sqlite_pipeline(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import environment_protection
        from core.pipeline import EventPipeline
        from core.security import SecurityAdapter
        from core.storage import load_session_events

        self.desktop.process_list = [
            {"pid": 20, "ppid": 1, "name": "msedge.exe", "create_time": 2},
            {"pid": 30, "ppid": 1, "name": "telegram.exe", "create_time": 3},
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = EventPipeline(root / "events.db", root / "shots")
            pipeline.start()
            adapter = SecurityAdapter("environment_protection", pipeline.submit)
            try:
                with patch.object(environment_protection, "_protection", self.protection):
                    self.assertTrue(adapter.enable()[0])
                    self.desktop.hook(key("print screen", 55))
                    self.assertTrue(wait_for(lambda: pipeline.current_risk().total > 0))
                    adapter.disable()
            finally:
                pipeline.stop()
            stored = load_session_events(root / "events.db", pipeline.session_id)
            self.assertEqual(len(stored), 2)
            by_type = {entry.event.type.value: entry.event for entry in stored}
            self.assertEqual(set(by_type), {"hotkey_blocked", "suspicious_process"})
            self.assertEqual(by_type["hotkey_blocked"].source, "security")
            self.assertEqual(
                {p["pid"] for p in by_type["suspicious_process"].details["processes"]}, {20, 30}
            )
            self.assertIsNone(self.desktop.hook)


if __name__ == "__main__":
    unittest.main()
