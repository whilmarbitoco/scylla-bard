import pyttsx3
import speech_recognition as sr
from transformers import AlbertTokenizer, AlbertForQuestionAnswering
import torch


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "Timeout: No speech detected"
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Error requesting results: {e}"


def speak(txt):
    engine = pyttsx3.init()
    engine.setProperty('rate', 125)
    engine.say(txt)
    engine.runAndWait()
    engine.stop()


def ai(question, context):
    # Load the pretrained ALBERT model and tokenizer
    model_name = "albert-base-v2"
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForQuestionAnswering.from_pretrained(model_name)

    # Tokenize the context and question
    inputs = tokenizer(context, question, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Interpret the model's output for question answering
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0][answer_start:answer_end+1]))

    return answer


def main():
    speak("Hey! I am Scylla, your personal AI assistant.")
    while True:
        user_input = speech_to_text()
        print("User:", user_input)
        if user_input.lower() == "exit":
            break
        question = "What can you tell me about this?"
        ai_response = ai(question, user_input)
        print("AI:", ai_response)
        speak(ai_response)


if __name__ == "__main__":
    main()
