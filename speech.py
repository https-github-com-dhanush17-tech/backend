import speech_recognition as speech_recog
rec = speech_recog.Recognizer()
import sys
from mcq import mcq

#read duration from the arguments
duration = 10
print("Please talk")
with speech_recog.Microphone() as source:
    # read the audio data from the default microphone
    audio_data = rec.record(source, duration=duration)
    print("Recognizing...")
    # convert speech to text
    text = rec.recognize_google(audio_data)
    print("Test:", text)
    print(mcq(text))

#
# with speech_recog.Microphone() as source:
#     audio_data = rec.record(source, duration=duration)

def parseAudio(audio):
    returnArr = []
    try:
        list = rec.recognize(audio,True)                  # generate a list of possible transcriptions
        print("Possible transcriptions:")
        for prediction in list:
            print(" " + prediction["text"] + " (" + str(prediction["confidence"]*100) + "%)")
        return list
    except LookupError:
        # speech is unintelligible
        return {"text": ["-1"], "confidence": ["-1"]}