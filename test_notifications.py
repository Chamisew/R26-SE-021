import os
import sys

# Load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

# Add the dashboard directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dashboard"))

print("=" * 60)
print("Testing Notification System")
print("=" * 60)

# Check environment variables
print("\n1. Checking Environment Variables:")
env_vars = [
    "DISCORD_WEBHOOK_URL",
    "SMTP_SERVER",
    "SMTP_PORT",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
    "EMAIL_RECIPIENTS"
]

for var in env_vars:
    value = os.environ.get(var, "")
    status = "✓ SET" if value else "✗ NOT SET"
    if var == "SMTP_PASSWORD" and value:
        value = "********"
    print(f"   {var}: {status} {value}")

# Test the notifier
print("\n2. Testing Notifications...")
try:
    from notifier import send_alarm_notifications
    
    test_payload = {
        "service": "test-service",
        "prob": 0.95,
        "cpu": 85.5,
        "ts": "2026-05-10T12:00:00"
    }
    
    print("\nSending test alarm notifications...")
    send_alarm_notifications(test_payload)
    print("\n✓ Test notifications sent! Check your Discord and Email.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
