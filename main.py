import pyttsx3
import speech_recognition as sr
import bardapi

class AIAssistant:
    def __init__(self):
        self.token = ''
        self.bard = bardapi.core.Bard(self.token)
        self.recognizer = sr.Recognizer()

    def speech_to_text(self):
        with sr.Microphone() as source:
            print("Say something...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                return "Timeout: No speech detected"
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Error requesting results: {e}"

    def speak(self, txt):
        engine = pyttsx3.init()
        engine.setProperty('rate', 125)
        engine.say(txt)
        engine.runAndWait()
        engine.stop()

    def run(self):
        self.speak("Hello, I am Scylla.")
        x = self.speech_to_text()
        print("User:", x)
        response = self.bard.get_answer(x)
        i = response.get('content', "No response from AI.")
        print("AI:", i)
        self.speak(i)

if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.run()
