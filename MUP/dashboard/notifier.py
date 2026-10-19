"""
Component 3 -- Desktop Notification Module
Uses plyer ONLY — win10toast is broken on Python 3.11 (classAtom AttributeError).
"""
import logging
logger = logging.getLogger(__name__)

def send_alert(service_name, project_id, memory_prob, mem_growth, ram_mean, timestamp):
    """Send desktop notification for a detected failure."""
    title = "🖥️ SYSTEM ALERT: Memory Leak Warning"
    msg = (
        f"Target Component: {service_name} ({project_id})\n"
        f"Criticality     : {int(memory_prob * 100)}% Probability\n"
        f"Growth Rate     : +{mem_growth:.4f} MB/s\n"
        f"Avg RAM Usage   : {ram_mean:.2f} MB\n"
        f"Logged At       : {timestamp}"
    )

    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=msg[:256],
            app_name="Component 3 Dashboard",
            timeout=8
        )
        logger.info(f"[notifier] Notification sent for {service_name}")
    except Exception as e:
        logger.warning(f"[notifier] Could not send notification: {e}")
