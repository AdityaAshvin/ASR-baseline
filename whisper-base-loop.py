from transformers import pipeline
import torch
import os
from evaluate import load
import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

audio_dir = ''
transcription_dir = ''

audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
batch_size = 50

if audio_files:
    all_predicted_transcriptions = []
    all_correct_transcriptions = []
    predicted_transcriptions_by_person = {"child": "", "psych": ""}
    correct_transcriptions_by_person = {"child": "", "psych": ""}
    skipped_files = []

    num_batches = math.ceil(len(audio_files) / batch_size)

    cnt = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(audio_files))
        batch_files = audio_files[start_idx:end_idx]

        for audio_file_name in batch_files:
            print("Iteration ", cnt, " / 353")
            cnt += 1
            audio_path = os.path.join(audio_dir, audio_file_name)
            transcription_path = os.path.join(transcription_dir, f"{os.path.splitext(audio_file_name)[0]}")

            if not os.path.exists(transcription_path):
                print("Transcription file for",  audio_file_name, " not found.")
                skipped_files.append(audio_file_name)
                continue

            print("Audio file: ", audio_file_name)

            result = pipe(audio_path, max_new_tokens=256, generate_kwargs={"task": "transcribe"}, chunk_length_s=30, batch_size=8, return_timestamps=True)
            chunks = result["chunks"]

            correct_transcriptions = []
            with open(transcription_path, "r") as file:
                for line in file:
                    line_split = line.strip().split()
                    person_name = line_split[0]
                    start_time = float(line_split[1])
                    end_time = float(line_split[2])
                    transcription = ' '.join(line_split[3:])
                    correct_transcriptions.append((person_name, start_time, end_time, transcription))

            skip_file = False
            for i, chunk in enumerate(chunks):
                chunk_start, chunk_end = chunk["timestamp"]
                chunk_text = chunk["text"]

                if chunk_start is None or chunk_end is None:
                    skip_file = True
                    break

                min_diff = float('inf')
                closest_segment = None
                for person_name, start_time, end_time, transcription in correct_transcriptions:
                    diff = abs((start_time + end_time) / 2 - (chunk_start + chunk_end) / 2)
                    if diff < min_diff:
                        min_diff = diff
                        closest_segment = (person_name, start_time, end_time, transcription)

                aligned_chunk_person, aligned_chunk_start, aligned_chunk_end, aligned_transcription = closest_segment
                chunk["person_name"] = aligned_chunk_person

            if skip_file:
                print(f"Skipping file {audio_file_name} due to missing timestamps.")
                skipped_files.append(audio_file_name)
                continue

            predicted_transcription_combined = ' '.join(chunk["text"].upper().replace(".", "").replace(",", "").replace("?", "") for chunk in chunks)

            correct_transcription_combined = ' '.join(transcription for _, _, _, transcription in correct_transcriptions)

            all_predicted_transcriptions.append(predicted_transcription_combined)
            all_correct_transcriptions.append(correct_transcription_combined)

            for person_name, start_time, end_time, transcription in correct_transcriptions:
                if person_name in correct_transcriptions_by_person:
                    correct_transcriptions_by_person[person_name] += " " + transcription

            for chunk in chunks:
                person_name = chunk["person_name"]
                if person_name in predicted_transcriptions_by_person:
                    predicted_transcriptions_by_person[person_name] += " " + chunk["text"].upper().replace(".", "").replace(",", "").replace("?", "")

    wer_metric = load("wer")

    combined_predicted_transcriptions = ' '.join(all_predicted_transcriptions)
    combined_correct_transcriptions = ' '.join(all_correct_transcriptions)

    wer = wer_metric.compute(references=[combined_correct_transcriptions], predictions=[combined_predicted_transcriptions])

    print("combined pred text check: ", combined_predicted_transcriptions[:1000] )
    print("combined corr text check: ", combined_correct_transcriptions[:1000] )

    print("combined pred child check: ", predicted_transcriptions_by_person['child'][:800])
    print("combined corr child check: ", correct_transcriptions_by_person['child'][:800])

    print("combined pred adult check: ", predicted_transcriptions_by_person['psych'][:800])
    print("combined corr adult check: ", correct_transcriptions_by_person['psych'][:800])

    print("word error rate", wer)

    wer_child_only = wer_metric.compute(references=[correct_transcriptions_by_person['child']], predictions=[predicted_transcriptions_by_person['child']])

    print("word error rate - only child speech", wer_child_only)

    wer_adult_only = wer_metric.compute(references=[correct_transcriptions_by_person['psych']], predictions=[predicted_transcriptions_by_person['psych']])

    print("word error rate - only adult speech", wer_adult_only)

    print("Skipped files:")
    for skipped_file in skipped_files:
        print(skipped_file)