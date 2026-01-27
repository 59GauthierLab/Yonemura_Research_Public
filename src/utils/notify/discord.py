from __future__ import annotations

import os

import requests


def send_discord_notification(message: str) -> bool:
    """
    Send a notification message to a Discord channel via webhook.

    This function is designed to be best-effort:
    any errors during notification will be caught and will not
    interrupt the main process.

    Args:
        message (str): The message to send.
    """

    discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not discord_webhook_url:
        # Environment misconfiguration should not stop long-running training
        return False

    data = {"content": message}

    # Do not interrupt the main process even if the notification fails
    try:
        response = requests.post(discord_webhook_url, json=data, timeout=5)
        if response.status_code in (200, 204):
            print("Discord notification sent successfully.")
            return True
        else:
            print(
                f"Failed to send Discord notification: "
                f"{response.status_code}, {response.text}"
            )
            return False
    except requests.RequestException as e:
        print(f"Network error while sending Discord notification: {e}")
        return False


def test_discord_notification() -> None:
    """
    Test function to send a sample notification.
    """
    success = send_discord_notification(
        "This is a test notification from the Discord webhook utility."
    )
    if success:
        print("Notification sent successfully.")
    else:
        print("Failed to send notification.")


if __name__ == "__main__":
    test_discord_notification()
