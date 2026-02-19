import pyautogui
import time

def execute_action(action):
    """Execute a desktop action based on string command."""
    if action == "play_pause":
        pyautogui.press("playpause")
    elif action == "next_track":
        pyautogui.press("nexttrack")
    elif action == "prev_track":
        pyautogui.press("prevtrack")
    elif action == "volume_up":
        pyautogui.press("volumeup")
    elif action == "volume_down":
        pyautogui.press("volumedown")
    elif action == "mute":
        pyautogui.press("volumemute")
    elif action == "tab_next":
        pyautogui.hotkey("ctrl", "tab")
    elif action == "tab_prev":
        pyautogui.hotkey("ctrl", "shift", "tab")
    elif action == "window_next":
        pyautogui.hotkey("alt", "tab")
    elif action == "window_prev":
        pyautogui.hotkey("alt", "shift", "tab")
    elif action == "browser_back":
        pyautogui.hotkey("alt", "left")
    elif action == "browser_forward":
        pyautogui.hotkey("alt", "right")
    # Add more as needed
    time.sleep(0.1)  # avoid rapid repeats