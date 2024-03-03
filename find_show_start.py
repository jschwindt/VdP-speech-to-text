#!/usr/bin/env -S python -u

import whisper
import sys
import re

lvst = re.compile("la venganza ser[aá] terrible", re.IGNORECASE)

model = whisper.load_model("small")
result = model.transcribe(sys.argv[1], verbose=None, language="es", fp16=False)

prev_text = ""
prev_time = 0
for segment in result['segments']:
    two_lines = prev_text + segment['text']
    two_lines_time = prev_time
    prev_time = segment['start']
    prev_text = segment['text']
    if lvst.search(segment['text']):
        print(segment['start'])
        sys.exit(0)
    if lvst.search(two_lines):
        print(two_lines_time)
        sys.exit(0)

sys.exit(1)
