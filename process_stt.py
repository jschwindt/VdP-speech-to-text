#!/usr/bin/env python

import logging
import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import click
import requests

import whisper


class TextProcessor:
    def __init__(self, uploader=None):
        self.uploader = uploader
        self.block_text = None
        self.block_time = None
        self.already_seen = {}

    def add(self, time, text):
        if not self.block_text:
            self.block_text = ""
            self.block_time = time if time > 0 else 1

        if self.already_seen.get(text):
            self.already_seen[text] += 1
            if self.already_seen[text] > 3:
                return
        else:
            self.already_seen[text] = 1

        self.block_text += f"{{{time}}} {text}\n"

        if (time - self.block_time) > 300:
            self.finish()

    def finish(self):
        if self.block_text:
            if self.uploader:
                upload_resp = self.uploader.upload_text(time={self.block_time}, text={self.block_text})
                logging.info(upload_resp)
            else:
                logging.info(f"NO uploader, time: {self.block_time}, text:\n{self.block_text}")
            self.block_text = None


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
    def __init__(self, model, text_processor=None):
        self.model = model
        self.text_processor = text_processor
        self.text = ""

    def recognize(self, audio_file):
        model = whisper.load_model(self.model)
        result = model.transcribe(audio_file, verbose=False, language="es", fp16=False)

        for segment in result['segments']:
            stripped_text = segment['text'].strip()
            self.text += stripped_text + "\n"
            if self.text_processor:
                self.text_processor.add(int(segment['start']), stripped_text)

        if self.text_processor:
            self.text_processor.finish()

        return self.text


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
@click.option('-m', '--model', default="small", type=click.Choice(['tiny', 'base', 'small', 'medium']),
              help='Model to use')
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
