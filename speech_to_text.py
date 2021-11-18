#!/Users/jschwindt/vosk-model-es/.venv/bin/python -u

import json
import subprocess

import click
from vosk import KaldiRecognizer, Model, SetLogLevel

sample_rate = 16000


def format_result(data):
    result = json.loads(data)
    text = result.get('text')
    if text and len(text) > 4:
        return f"{int(result['result'][0]['start'])}|{text}"


def reconize(model_path, process):
    vosk_model = Model(model_path)
    reconizer = KaldiRecognizer(vosk_model, sample_rate)
    reconizer.SetWords(True)

    while True:
        data = process.stdout.read(8000)
        if len(data) == 0:
            break
        if reconizer.AcceptWaveform(data):
            yield format_result(reconizer.Result())

    yield format_result(reconizer.FinalResult())


@click.command()
@click.option('-m', '--model', default="model", type=click.Path(exists=True),
              help='Path to the model')
@click.argument('audio-file', required=True, type=click.Path(exists=True))
@click.argument('output_text_file', required=False,
                type=click.Path(exists=False))
@click.option('-v', '--verbose', is_flag=True, help="Verbose information")
def speech_to_text(model, audio_file, output_text_file, verbose):
    SetLogLevel(-1)

    process = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "quiet", "-i", audio_file,
            "-ar", str(sample_rate), "-ac", "1", "-f", "s16le", "-",
        ],
        stdout=subprocess.PIPE,
    )

    output_f = None
    if output_text_file:
        output_f = open(output_text_file, mode="w")
    else:
        verbose = True

    for result in reconize(model, process):
        if result:
            if output_f:
                output_f.write(f"{result}\n")
            if verbose:
                print(result)

    if output_f:
        output_f.close()


if __name__ == '__main__':
    speech_to_text()
