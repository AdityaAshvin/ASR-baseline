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
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-medium", device=device
)

audio_dir = ''
transcription_dir = ''

audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
if audio_files:
    audio_file_name = audio_files[16]
    audio_path = os.path.join(audio_dir, audio_file_name)
    transcription_path = os.path.join(transcription_dir, f"{os.path.splitext(audio_file_name)[0]}")

    if not os.path.exists(transcription_path):
        print("Transcription file for",  audio_file_name, " not found.")
    else:
        print("Audio file: ", audio_file_name)

        result = pipe(audio_path, max_new_tokens=256, generate_kwargs={"task": "transcribe"}, chunk_length_s=30, batch_size=8, return_timestamps=True)
        chunks = result["chunks"]

    # Load the correct transcription
    correct_transcriptions = []
    with open(transcription_path, "r") as file:
        for line in file:
            line_split = line.strip().split()
            person_name = line_split[0]
            start_time = float(line_split[1])
            end_time = float(line_split[2])
            transcription = ' '.join(line_split[3:])
            correct_transcriptions.append((person_name, start_time, end_time, transcription))

    print("Predicted transcription with aligned start and end times:")
    for i, chunk in enumerate(chunks):
        chunk_start, chunk_end = chunk["timestamp"]
        chunk_text = chunk["text"]

        # Find the closest correct transcription segment
        min_diff = float('inf')
        closest_segment = None
        for person_name, start_time, end_time, transcription in correct_transcriptions:
            diff = abs((start_time + end_time) / 2 - (chunk_start + chunk_end) / 2)
            if diff < min_diff:
                min_diff = diff
                closest_segment = (person_name, start_time, end_time, transcription)

        # Align chunk start and end times with the closest correct transcription segment
        aligned_chunk_person, aligned_chunk_start, aligned_chunk_end, aligned_transcription = closest_segment
        chunk["person_name"] = aligned_chunk_person
        print(f"Chunk {i+1}: Person name: {aligned_chunk_person}, Start time: {aligned_chunk_start}, End time: {aligned_chunk_end}, Transcription: {chunk_text}")

    print(f"\nCorrect transcription segments:")
    for i, (person_name,  start_time, end_time, transcription) in enumerate(correct_transcriptions):
        print(f"Segment {i+1}: Person name: {person_name}, Start time: {start_time}, End time: {end_time}, Transcription: {transcription}")

    # Merge all timestamp-wise chunks into one big chunk for the predicted transcription
    predicted_transcription_combined = ' '.join(chunk["text"].upper().replace(".", "").replace(",", "").replace("?", "").replace("!", "") for chunk in chunks)

    # Merge all timestamp-wise chunks into one big chunk for the correct transcription
    correct_transcription_combined = ' '.join(transcription for _, _, _, transcription in correct_transcriptions)

    # Compute Word Error Rate
    wer_metric = load("wer")

    wer = wer_metric.compute(references=[correct_transcription_combined], predictions=[predicted_transcription_combined])

    print("combined chunk pred", predicted_transcription_combined)
    print("combined chunk correct", correct_transcription_combined)

    print("word error rate", wer)

    #print("predicted chunks: ", chunks)
    #print("correct segments: ", correct_transcriptions)

    # Create dictionaries to store transcriptions for each person
    predicted_transcriptions_by_person = {"child": "", "psych": ""}
    correct_transcriptions_by_person = {"child": "", "psych": ""}

    # Iterate through correct transcriptions and group them by person name
    for person_name, start_time, end_time, transcription in correct_transcriptions:
        if person_name in correct_transcriptions_by_person:
            correct_transcriptions_by_person[person_name] += " " + transcription

    # Iterate through predicted transcriptions and group them by person name
    for chunk in chunks:
        person_name = chunk["person_name"]  # Using the person name from the chunk data
        if person_name in predicted_transcriptions_by_person:
            predicted_transcriptions_by_person[person_name] += " " + chunk["text"].upper().replace(".", "").replace(",", "").replace("?", "").replace("!", "")

    # Print transcriptions by person
    print(f"Prediction (child): {predicted_transcriptions_by_person['child']}")
    print(f"Correct (child): {correct_transcriptions_by_person['child']}")

    print(f"Prediction (psych): {predicted_transcriptions_by_person['psych']}")
    print(f"Correct (psyc): {correct_transcriptions_by_person['psych']}")

    wer_child_only = wer_metric.compute(references=[correct_transcriptions_by_person['child']], predictions=[predicted_transcriptions_by_person['child']])

    print("word error rate - only child speech", wer_child_only)

    wer_adult_only = wer_metric.compute(references=[correct_transcriptions_by_person['psych']], predictions=[predicted_transcriptions_by_person['psych']])

    print("word error rate - only adult speech", wer_adult_only)
