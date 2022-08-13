#!/usr/bin/env python

import json
import logging
import os
import subprocess
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import click
import requests
from vosk import KaldiRecognizer, Model, SetLogLevel


class TextBlock:
    def __init__(self):
        self.block_time = None
        self.line_time = None
        self.text = ''

    def add(self, time, text):
        if self.block_time is None:
            self.block_time = time
            self.line_time = time
            self.text += f"{{{time}}} {text}"
            return time - self.block_time

        if (time - self.line_time) <= 10:
            self.text += f" {text}"
        else:
            self.text += f"\n{{{time}}} {text}"
            self.line_time = time
        return time - self.block_time

    def close(self):
        self.text += "\n"


class TextProcessor:
    def __init__(self, uploader=None):
        self.text_block = None
        self.uploader = uploader

    def add(self, time, text):
        if not self.text_block:
            self.text_block = TextBlock()

        block_duration = self.text_block.add(time, text)
        if block_duration > 300:
            self.finish()

    def finish(self):
        if self.text_block:
            self.text_block.close()
            if self.uploader:
                upload_resp = self.uploader.upload_text(time={self.text_block.block_time}, text={self.text_block.text})
                logging.info(upload_resp)
            else:
                logging.info(f"NO uploader, time: {self.text_block.block_time}, text:\n{self.text_block.text}")
            self.text_block = None


@contextmanager
def download_to_tmp(url):
    tmp_file = NamedTemporaryFile(delete=False)
    logging.info(f"Created tmp: {tmp_file.name} for {url}")
    try:
        resp = requests.get(url, stream=True)
        for chunk in resp.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_file.close()
        logging.info("Donwload complete!")
        yield tmp_file.name
    finally:
        os.unlink(tmp_file.name)
        logging.info(f"Deleted tmp: {tmp_file.name}")


class SpeechToText:
    def __init__(self, model_path, text_processor=None):
        SetLogLevel(-1)
        self.vosk_model = Model(model_path)
        self.text_processor = text_processor
        self.sample_rate = 16000

    def recognize(self, audio_file):
        process = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "quiet", "-i", audio_file,
                "-ar", str(self.sample_rate), "-ac", "1", "-f", "s16le", "-",
            ],
            stdout=subprocess.PIPE,
        )
        self.text = ""
        for result in self.next_sentence(process):
            if result:
                self.text += result + "\n"
                logging.info(result)

        return self.text

    def format_result(self, data, final=False):
        result = json.loads(data)
        text = result.get('text')
        if text:
            time = int(result['result'][0]['start'])
            if self.text_processor:
                self.text_processor.add(time, text)
        if final:
            self.text_processor.finish()
        if text:
            return f"{time}|{text}"

    def next_sentence(self, process):
        reconizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
        reconizer.SetWords(True)
        while True:
            data = process.stdout.read(8000)
            if len(data) == 0:
                break
            if reconizer.AcceptWaveform(data):
                yield self.format_result(reconizer.Result())
        yield self.format_result(reconizer.FinalResult(), final=True)


class VdpApi:
    BASE_URL = os.getenv('VDP_STT_URL', "http://localhost:3000/speech_to_text")
    TOKEN = os.getenv('VDP_STT_TOKEN', "")
    HEADERS = {
        "Accept": "application/json",
        "Authorization": TOKEN,
    }

    def __init__(self):
        self.audio = None

    def next_audio_url(self):
        resp = requests.get(f"{VdpApi.BASE_URL}/next", headers=VdpApi.HEADERS)
        if resp.status_code != 200:
            logging.error(f"next_audio_url status: {resp.status_code}")
            return
        audios = resp.json()
        if audios:
            self.audio = audios[0]
            return self.audio["url"]

    def start(self):
        resp = requests.put(f"{VdpApi.BASE_URL}/start/{self.audio['id']}", headers=VdpApi.HEADERS)
        if resp.status_code != 200:
            logging.error(f"start status: {resp.status_code}")
            return
        return resp.json()

    def upload_text(self, time, text):
        resp = requests.put(
            f"{VdpApi.BASE_URL}/update/{self.audio['id']}",
            headers=VdpApi.HEADERS,
            data={
                "time": time,
                "text": text,
            },
        )
        if resp.status_code != 200:
            logging.error(f"upload_text status: {resp.status_code}")
            return
        return resp.json()


@click.command()
@click.option('-m', '--model', default="model", type=click.Path(exists=True),
              help='Path to the model')
@click.argument('audio-file', required=False)
def main(model, audio_file):
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    if audio_file:
        text_procesor = TextProcessor()
        recognizer = SpeechToText(model, text_procesor)
        text = recognizer.recognize(audio_file)
        logging.info(f"RESULT:\n{text}")
    else:
        vdp = VdpApi()
        text_procesor = TextProcessor(uploader=vdp)
        recognizer = SpeechToText(model, text_procesor)
        while True:
            audio_url = vdp.next_audio_url()
            if not audio_url:
                break
            start_resp = vdp.start()
            logging.info(start_resp)
            with download_to_tmp(audio_url) as audio_file:
                text = recognizer.recognize(audio_file)


if __name__ == "__main__":
    main()
