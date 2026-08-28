"""
SRE Command Center — Windows System Notifier
============================================
Run this alongside server.py and the React dev server.
It polls the dashboard API every 3 seconds and fires real Windows
system-tray balloon notifications + system sounds whenever a service
enters WARNING or CRITICAL state.

Usage:
    python notifier.py

Requirements: Python 3.x on Windows (uses only built-in modules)
"""

import subprocess
import sys
import json
import time
import threading
import base64
import urllib.request
import urllib.error
from datetime import datetime

try:
    import winreg
except ImportError:
    winreg = None

# Force UTF-8 output to prevent codec errors on Windows CP1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Config ──────────────────────────────────────────────────────────────────
API_URL      = "http://127.0.0.1:8766/api/dashboard"
POLL_SEC     = 3            # seconds between API polls
CRIT_REPEAT  = 15           # re-alert active CRITICAL services every 15s
WARN_REPEAT  = 30           # re-alert active WARNING services every 30s

# ── State tracking ──────────────────────────────────────────────────────────
_last_risk: dict[str, str] = {}      # service_name -> last risk state
_last_time: dict[str, float] = {}    # service_name -> timestamp of last system alert


# ── Register Custom System App ID in Windows Registry ───────────────────────
def ensure_system_aumid_registered():
    """
    Registers 'SentinelAIOps' in HKCU Registry so Windows Toast Notifications
    display 'Sentinel AIOps' as the sender app name instead of 'Windows PowerShell'.
    """
    if not winreg or sys.platform != "win32":
        return
    try:
        key_path = r"Software\Classes\AppUserModelId\SentinelAIOps"
        key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path)
        winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Sentinel AIOps")
        winreg.CloseKey(key)
    except Exception as e:
        print(f"[notifier] Registry registration notice: {e}", flush=True)


# ── Windows System Sound ──────────────────────────────────────────────────
def play_system_sound(risk: str):
    """Play a Windows system sound appropriate for the risk level."""
    sound_map = {
        "CRITICAL": "SystemHand",          # Windows Critical sound
        "WARNING":  "SystemExclamation",   # Windows Exclamation sound
    }
    alias = sound_map.get(risk, "SystemDefault")
    try:
        import winsound
        winsound.PlaySound(alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
    except Exception as e:
        print(f"[notifier] Sound error: {e}", flush=True)


# ── Windows System Desktop Notification (TopMost Alert Window + Toast) ──────
def send_windows_toast(title: str, message: str, risk: str):
    """
    Fire a TopMost visual desktop alert window + Windows Toast under the 'Sentinel AIOps' brand.
    Guarantees 100% visual visibility over all open windows regardless of OS focus settings.
    """
    ensure_system_aumid_registered()
    scenario = "alarm" if risk == "CRITICAL" else "reminder"
    
    # Color indicators (Red for CRITICAL, Yellow for WARNING, Green for HEALTHY)
    if risk == "CRITICAL":
        accent_rgb = "224, 57, 62"
        sound_cmd = "[System.Media.SystemSounds]::Hand.Play()"
    elif risk == "WARNING":
        accent_rgb = "228, 140, 0"
        sound_cmd = "[System.Media.SystemSounds]::Exclamation.Play()"
    else:
        accent_rgb = "26, 135, 84"
        sound_cmd = "[System.Media.SystemSounds]::Asterisk.Play()"

    title_safe   = title.replace('"', '`"').replace("'", "`'").replace("&", "&amp;").replace("<", "&lt;")
    message_safe = message.replace('"', '`"').replace("'", "`'").replace('\n', '`n').replace("&", "&amp;").replace("<", "&lt;")

    ps_script = f"""
$ErrorActionPreference = 'SilentlyContinue'

# 1. Try Windows Runtime Toast Notification for Action Center logging
try {{
    [void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
    [void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
    [void][Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType = WindowsRuntime]

    $xml = @"
<toast scenario="{scenario}" duration="long">
  <visual>
    <binding template="ToastGeneric">
      <text>{title_safe}</text>
      <text>{message_safe}</text>
    </binding>
  </visual>
</toast>
"@
    $xdoc = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xdoc.LoadXml($xml)
    $toast = [Windows.UI.Notifications.ToastNotification]::new($xdoc)
    $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('SentinelAIOps')
    $notifier.Show($toast)
}} catch {{}}

# 2. Render Un-suppressible TopMost Visual Alert Popup Window
try {{
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Sentinel AIOps Alert"
    $form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::None
    $form.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
    $form.Width = 390
    $form.Height = 100
    $form.TopMost = $true
    $form.ShowInTaskbar = $false
    $form.BackColor = [System.Drawing.Color]::FromArgb(20, 24, 32)

    # Position at bottom-right of primary screen
    $screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
    $form.Left = $screen.Right - 405
    $form.Top = $screen.Bottom - 115

    # Accent color bar on left edge
    $bar = New-Object System.Windows.Forms.Label
    $bar.Width = 6
    $bar.Height = 100
    $bar.Left = 0
    $bar.Top = 0
    $bar.BackColor = [System.Drawing.Color]::FromArgb({accent_rgb})
    $form.Controls.Add($bar)

    # Header Title
    $titleLabel = New-Object System.Windows.Forms.Label
    $titleLabel.Text = "{title_safe}"
    $titleLabel.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
    $titleLabel.ForeColor = [System.Drawing.Color]::White
    $titleLabel.Left = 16
    $titleLabel.Top = 10
    $titleLabel.AutoSize = $true
    $form.Controls.Add($titleLabel)

    # Message Body
    $msgLabel = New-Object System.Windows.Forms.Label
    $msgLabel.Text = "{message_safe}"
    $msgLabel.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $msgLabel.ForeColor = [System.Drawing.Color]::FromArgb(190, 198, 210)
    $msgLabel.Left = 16
    $msgLabel.Top = 36
    $msgLabel.Width = 360
    $msgLabel.Height = 55
    $form.Controls.Add($msgLabel)

    # Play system sound at exact millisecond window is rendered
    {sound_cmd}

    $form.Show()

    # Auto close after 7 seconds
    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 7000
    $timer.add_Tick({{
        $form.Close()
    }})
    $timer.Start()

    [System.Windows.Forms.Application]::Run($form)
}} catch {{}}
"""
    try:
        encoded_ps = base64.b64encode(ps_script.encode('utf-16le')).decode('utf-8')
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-EncodedCommand", encoded_ps],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except Exception as e:
        print(f"[notifier] Notification dispatch error: {e}", flush=True)


# ── Alert dispatcher ──────────────────────────────────────────────────────
def dispatch_alert(service: str, risk: str, action: str):
    """Send system notification in a background thread (sound is produced only by the notification)."""
    if risk == "CRITICAL":
        title   = f"[CRITICAL ALARM] -- {service}"
        message = (
            f"Service '{service}' has reached CRITICAL failure probability!\n"
            f"Action: {action.replace('_', ' ')}"
        )
    else:
        title   = f"[SYSTEM WARNING] -- {service}"
        message = (
            f"Service '{service}' has elevated risk -- monitoring required.\n"
            f"Action: {action.replace('_', ' ')}"
        )

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{risk}] {title}", flush=True)
    print(f"         {message}", flush=True)
    print(flush=True)

    threading.Thread(target=lambda: send_windows_toast(title, message, risk), daemon=True).start()


# ── Recovery notification ─────────────────────────────────────────────────
def dispatch_recovery(service: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [RECOVERED] {service} is back to HEALTHY", flush=True)
    title   = f"[RECOVERED] -- {service}"
    message = f"Service '{service}' has returned to normal healthy operation."

    threading.Thread(target=lambda: send_windows_toast(title, message, "HEALTHY"), daemon=True).start()


# ── Main polling loop ─────────────────────────────────────────────────────
def poll():
    global _last_risk, _last_time
    print(f"[notifier] SRE System Alert Monitor started")
    print(f"[notifier] Polling {API_URL} every {POLL_SEC}s …")
    print(f"[notifier] System Toast + System Tray notifications active\n", flush=True)

    consecutive_errors = 0

    while True:
        try:
            with urllib.request.urlopen(API_URL, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            consecutive_errors = 0
        except urllib.error.URLError:
            consecutive_errors += 1
            if consecutive_errors == 1:
                print(f"[notifier] Cannot reach {API_URL} — is server.py running?", flush=True)
            time.sleep(POLL_SEC)
            continue
        except Exception as e:
            print(f"[notifier] Poll error: {e}", flush=True)
            time.sleep(POLL_SEC)
            continue

        services = data.get("services", [])
        now = time.time()

        for svc in services:
            name   = svc.get("service", "")
            risk   = svc.get("risk", "HEALTHY")
            action = svc.get("action", "NO_ACTION")

            if risk in ("CRITICAL", "WARNING"):
                prev_risk = _last_risk.get(name)
                last_t    = _last_time.get(name, 0)
                repeat_sec = CRIT_REPEAT if risk == "CRITICAL" else WARN_REPEAT

                # Fire alert if: state changed OR repeat interval elapsed
                if prev_risk != risk or (now - last_t) >= repeat_sec:
                    dispatch_alert(name, risk, action)
                    _last_risk[name] = risk
                    _last_time[name] = now
            else:
                # Service recovered from WARNING/CRITICAL
                if name in _last_risk:
                    dispatch_recovery(name)
                    del _last_risk[name]
                    _last_time.pop(name, None)

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        poll()
    except KeyboardInterrupt:
        print("\n[notifier] Stopped.", flush=True)

