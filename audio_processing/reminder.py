import time
import threading
from datetime import datetime, timedelta
import pyttsx3
from PyQt5.QtCore import QObject, pyqtSignal

class ReminderSystem(QObject):
    reminder_triggered = pyqtSignal(str)  # Signal to notify UI of reminder trigger
    reminder_updated = pyqtSignal()  # Signal to notify UI to update list

    def __init__(self):
        super().__init__()
        self.reminders = []
        self.running = False
        self.reminder_thread = None
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS engine
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Get available voices and set a female voice if available
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if "female" in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
    def add_reminder(self, task_name, interval_minutes):
        """Add a new reminder with task name and interval"""
        reminder = {
            'task_name': task_name,
            'interval': interval_minutes,
            'last_triggered': datetime.now(),
            'next_trigger': datetime.now() + timedelta(minutes=interval_minutes),
            'active': True
        }
        self.reminders.append(reminder)
        self.reminder_updated.emit()
        return reminder
        
    def remove_reminder(self, task_name):
        """Remove a reminder by task name"""
        self.reminders = [r for r in self.reminders if r['task_name'] != task_name]
        self.reminder_updated.emit()
        
    def toggle_reminder(self, task_name):
        """Toggle a reminder's active state"""
        for reminder in self.reminders:
            if reminder['task_name'] == task_name:
                reminder['active'] = not reminder['active']
                if reminder['active']:
                    # Reset the next trigger time when reactivating
                    reminder['next_trigger'] = datetime.now() + timedelta(minutes=reminder['interval'])
        self.reminder_updated.emit()
        
    def get_reminders(self):
        """Get all reminders"""
        return self.reminders
        
    def announce_task(self, task_name):
        """Announce the task using text-to-speech"""
        try:
            # Create announcement text
            announcement = f"Time to {task_name}!"
            
            # Use a separate thread for TTS to avoid blocking
            def speak():
                self.tts_engine.say(announcement)
                self.tts_engine.runAndWait()
            
            threading.Thread(target=speak, daemon=True).start()
            
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
        
    def start(self):
        """Start the reminder system"""
        self.running = True
        self.reminder_thread = threading.Thread(target=self._check_reminders)
        self.reminder_thread.daemon = True
        self.reminder_thread.start()
        
    def stop(self):
        """Stop the reminder system"""
        self.running = False
        if self.reminder_thread:
            self.reminder_thread.join()
        self.tts_engine.stop()
            
    def _check_reminders(self):
        """Background thread to check and trigger reminders"""
        while self.running:
            current_time = datetime.now()
            
            for reminder in self.reminders:
                if reminder['active'] and current_time >= reminder['next_trigger']:
                    # Announce the task
                    self.announce_task(reminder['task_name'])
                    
                    # Emit signal for UI update
                    self.reminder_triggered.emit(reminder['task_name'])
                    
                    # Update reminder timing
                    reminder['last_triggered'] = current_time
                    reminder['next_trigger'] = current_time + timedelta(minutes=reminder['interval'])
                    self.reminder_updated.emit()
            
            # Check every second
            time.sleep(1) 