from datasets import load_dataset
from transformers import pipeline
import torch
import numpy as np
import jiwer
from evaluate import load

from pydub import AudioSegment
from pydub.playback import play
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline( "automatic-speech-recognition", model="facebook/wav2vec2-large-960h-lv60-self",device=device )

audio_dir = ''
transcription_dir = ''

audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
if audio_files:
    audio_file_name = audio_files[0]
    audio_path = os.path.join(audio_dir, audio_file_name)
    transcription_path = os.path.join(transcription_dir, f"{os.path.splitext(audio_file_name)[0]}")

    # Check if the transcription file exists
    if not os.path.exists(transcription_path):
        print("Transcription file for",  audio_file_name, " not found.")
    else:
        print("Audio file: ", audio_file_name)

        #result = pipe(audio_path)
        predicted_transcription_pipe = pipe(audio_path,generate_kwargs={"task": "transcribe"},
            chunk_length_s=30, batch_size=8, return_timestamps = "word")
        predicted_transcription = predicted_transcription_pipe['text']

        correct_transcriptions = []
        with open(transcription_path, "r") as file:
            for line in file:
                line_split = line.strip().split()
                person_name = line_split[0]
                start_time = float(line_split[1])
                end_time = float(line_split[2])
                transcription = ' '.join(line_split[3:])
                correct_transcriptions.append((person_name, start_time, end_time, transcription))

    correct_transcription_combined = ' '.join(transcription for _, _, _, transcription in correct_transcriptions)

    wer_metric = load("wer")

    wer = wer_metric.compute(references=[correct_transcription_combined], predictions=[predicted_transcription])

    print("combined chunk pred", predicted_transcription)
    print("combined chunk correct", correct_transcription_combined)

    print("word error rate", wer)

    predicted_transcription_with_timestamps = predicted_transcription_pipe["chunks"]

    predicted_transcriptions_by_person = {'child': [], 'psych': []}
    correct_transcriptions_by_person = {'child': [], 'psych': []}

    for word_info in predicted_transcription_with_timestamps:
        word = word_info['text']
        start_time, end_time = word_info['timestamp']
        aligned_person = None
        aligned_transcription_segment = None

        for person_name, person_start, person_end, transcription in correct_transcriptions:
            if start_time >= person_start and end_time <= person_end:
                aligned_person = person_name
                aligned_transcription_segment = transcription
                break

        if aligned_person:
            predicted_transcriptions_by_person[aligned_person].append((start_time, end_time, word))

    combined_transcriptions_by_person = {'child': [], 'psych': []}
    for person_name in correct_transcriptions_by_person.keys():
        current_segment = None
        current_text = []

        for start_time, end_time, word in predicted_transcriptions_by_person[person_name]:
            if current_segment != word:
                if current_text:
                    combined_transcriptions_by_person[person_name].append(' '.join(current_text))
                current_text = [word]
                current_segment = word
            else:
                current_text.append(word)

        if current_text:
            combined_transcriptions_by_person[person_name].append(' '.join(current_text))

    predicted_transcription_child = ' '.join(combined_transcriptions_by_person['child'])
    predicted_transcription_psych = ' '.join(combined_transcriptions_by_person['psych'])

    correct_transcription_child = ' '.join([t for p, s, e, t in correct_transcriptions if p == 'child'])
    correct_transcription_psych = ' '.join([t for p, s, e, t in correct_transcriptions if p == 'psych'])

    print("Combined Predicted Transcription (Child): ", predicted_transcription_child)
    print("Combined Correct Transcription (Child): ", correct_transcription_child)
    print("Combined Predicted Transcription (Psych): ", predicted_transcription_psych)
    print("Combined Correct Transcription (Psych): ", correct_transcription_psych)

    wer_child_only = wer_metric.compute(references=[correct_transcription_child], predictions=[predicted_transcription_child])
    wer_adult_only = wer_metric.compute(references=[correct_transcription_psych], predictions=[predicted_transcription_psych])

    print("wer child only: ", wer_child_only)
    print("wer adult only: ", wer_adult_only)