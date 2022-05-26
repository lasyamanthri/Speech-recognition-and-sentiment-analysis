from sklearn import pipeline
import speech_recognition
import pyttsx3
import pickle


pipeline = pickle.load(open('trained_model.sav','rb'))

recognizer = speech_recognition.Recognizer()

while True:
    try:

        with speech_recognition.Microphone() as mic:

            recognizer.adjust_for_ambient_noise(mic, duration=0.4)
            audio = recognizer.listen(mic)

            text = recognizer.recognize_google(audio, language = 'en-IN', show_all = True)
            #text = text.lower()

            print(f"Recognized {text['alternative'][0]['transcript']}")

            l = [text['alternative'][0]['transcript']]
            b = pipeline.predict(l)
            if b[0]==0:
                print("The user gave a negative statement")
            else:
                print("The user gave a positive statement")

            print(f"Confidence {text['alternative'][0]['confidence']}")
            break

            #print(text)

        
    except speech_recognition.UnknownValueError() as mic :

        recognizer.adjust_for_ambient_noise(mic, duration=0.2)
        continue
print("Thank you!")

