from transformers import pipeline
import torch
import os
from evaluate import load

audio_dir = ''
transcription_dir = ''

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model="patrickvonplaten/wavlm-libri-clean-100h-large", device=device)

wer_metric = load("wer")

audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

combined_pred = []
combined_pred_child = []
combined_pred_adult = []

combined_corr = []
combined_corr_child = []
combined_corr_adult = []

cnt = 1
for audio_file_name in audio_files:
    print("iteration: ", cnt, "/ 353")
    cnt += 1
    audio_path = os.path.join(audio_dir, audio_file_name)
    transcription_path = os.path.join(transcription_dir, os.path.splitext(audio_file_name)[0])

    if not os.path.exists(transcription_path):
        print("Transcription file for",  audio_file_name, " not found.")
        continue
    print("Audio file: ", audio_file_name)

    predicted_transcription_pipe = pipe(audio_path, generate_kwargs={"task": "transcribe"}, chunk_length_s=30, batch_size=8, return_timestamps="word")
    predicted_transcription = predicted_transcription_pipe['text'].upper()
    predicted_transcription_with_timestamps = predicted_transcription_pipe["chunks"]

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
    combined_pred.append(predicted_transcription)
    combined_corr.append(correct_transcription_combined)

    wer = wer_metric.compute(references=[correct_transcription_combined], predictions=[predicted_transcription])
    print("word error rate", wer)

    predicted_transcriptions_by_person = {'child': [], 'psych': []}
    for word_info in predicted_transcription_with_timestamps:
        word = word_info['text'].upper()
        start_time, end_time = word_info['timestamp']
        for person_name, person_start, person_end, _ in correct_transcriptions:
            if start_time >= person_start and end_time <= person_end:
                if person_name in predicted_transcriptions_by_person:
                    predicted_transcriptions_by_person[person_name].append(word)
                break

    predicted_transcription_child = ' '.join(predicted_transcriptions_by_person['child'])
    predicted_transcription_psych = ' '.join(predicted_transcriptions_by_person['psych'])

    correct_transcription_child = ' '.join([t for p, _, _, t in correct_transcriptions if p == 'child'])
    correct_transcription_psych = ' '.join([t for p, _, _, t in correct_transcriptions if p == 'psych'])

    combined_pred_child.append(predicted_transcription_child)
    combined_corr_child.append(correct_transcription_child)

    combined_pred_adult.append(predicted_transcription_psych)
    combined_corr_adult.append(correct_transcription_psych)

combined_pred_text = ' '.join(combined_pred)
combined_corr_text = ' '.join(combined_corr)

combined_pred_child_text = ' '.join(combined_pred_child)
combined_corr_child_text = ' '.join(combined_corr_child)

combined_pred_adult_text = ' '.join(combined_pred_adult)
combined_corr_adult_text = ' '.join(combined_corr_adult)

print("combined pred text check: ", combined_pred_text[:1000] )
print("combined corr text check: ", combined_corr_text[:1000] )

print("combined pred child check: ", combined_pred_child_text[:800])
print("combined corr child check: ", combined_corr_child_text[:800])

print("combined pred adult check: ", combined_pred_adult_text[:800])
print("combined corr adult check: ", combined_corr_adult_text[:800])

wer_combined_overall = wer_metric.compute(references=[combined_corr_text], predictions=[combined_pred_text])
wer_child_only = wer_metric.compute(references=[combined_corr_child_text], predictions=[combined_pred_child_text])
wer_adult_only = wer_metric.compute(references=[combined_corr_adult_text], predictions=[combined_pred_adult_text])

print("Overall combined WER: ", wer_combined_overall)
print("Child WER: ", wer_child_only)
print("Adult WER: ", wer_adult_only)