import os
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger("notifier")

def send_windows_toast(title, message):
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            title,
            message,
            duration=10,
            threaded=True
        )
        logger.info(f"Windows Toast sent: {title}")
    except ImportError:
        logger.warning("win10toast not installed, skipping Windows Toast")
    except Exception as e:
        logger.error(f"Failed to send Windows Toast: {e}")

def send_discord_webhook(message):
    try:
        import requests
        
        DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
        
        if not DISCORD_WEBHOOK_URL:
            logger.warning("DISCORD_WEBHOOK_URL not set, skipping Discord notification")
            return
        
        payload = {
            "content": message,
            "username": "CPU Spike Alert Bot",
            "avatar_url": "https://i.imgur.com/4M34hi2.png"
        }
        
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Discord notification sent successfully")
    except ImportError:
        logger.warning("requests not available, skipping Discord notification")
    except Exception as e:
        logger.error(f"Failed to send Discord notification: {e}")

def send_email_alert(subject, body):
    try:
        SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
        SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
        SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
        EMAIL_RECIPIENTS = os.environ.get("EMAIL_RECIPIENTS", "")
        
        if not SMTP_USERNAME or not SMTP_PASSWORD or not EMAIL_RECIPIENTS:
            logger.warning("Email credentials not set, skipping email alert")
            return
        
        recipients = [email.strip() for email in EMAIL_RECIPIENTS.split(",")]
        
        msg = MIMEMultipart()
        msg["From"] = SMTP_USERNAME
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        
        msg.attach(MIMEText(body, "plain"))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        text = msg.as_string()
        server.sendmail(SMTP_USERNAME, recipients, text)
        server.quit()
        
        logger.info(f"Email alert sent successfully to {len(recipients)} recipients")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")

def send_alarm_notifications(alarm_payload):
    service = alarm_payload["service"]
    prob = alarm_payload["prob"]
    cpu = alarm_payload["cpu"]
    ts = alarm_payload["ts"]
    
    title = f"⚠️ CPU SPIKE ALARM: {service}"
    message = (f"Service: {service}\n"
               f"Probability: {prob:.3f}\n"
               f"CPU Usage: {cpu:.1f}%\n"
               f"Time: {ts}")
    
    send_windows_toast(title, message)
    send_discord_webhook(f"⚠️ **CPU SPIKE ALERT**\n{message}")
    send_email_alert(title, message)
