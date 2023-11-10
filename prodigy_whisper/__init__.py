import tempfile 
from pathlib import Path 
from copy import deepcopy
from typing import List, Optional

from pydub import AudioSegment
from prodigy.components.preprocess import fetch_media as fetch_media_preprocessor
from prodigy.components.preprocess import fetch_task_media
from prodigy.components.stream import get_stream
from prodigy.core import Arg, recipe
from prodigy.protocols import ControllerComponentsDict
from prodigy.types import SourceType, StreamType, TaskType
from prodigy.util import log, set_hashes


import whisper
from whisper.model import Whisper

def add_annotations(stream, model: Whisper, model_name: str, segment: bool=False):
    for ex in stream:
        ex['meta']['model_name'] = model_name
        # If you don't pass fp16 you'll get an annoying warning 
        result = model.transcribe(ex['path'], fp16=False)
        if segment:
            log(f"RECIPE: {ex['meta']['path']} has {len(result['segments'])} segments.")
            for seg in result['segments']:
                # Make sure we don't override original example
                example_segment = deepcopy(ex)
                example_segment['transcript'] = seg['text']
                example_segment['orig_transcript'] = seg['text']
                example_segment['meta']['start'] = seg['start']
                example_segment['meta']['end'] = seg['end']
                example_segment['meta']['segment_id'] = seg['id']

                # To keep things simple, write to a temporary file locally
                # This ensures we can keep using same utility functions
                audio = AudioSegment.from_file(example_segment['meta']['path'])
                with tempfile.TemporaryDirectory("+wb") as tmpdir:
                    out_file = str(Path(tmpdir) / "out.mp3")
                    audio[seg['start'] * 1000: seg['end'] * 1000].export(out_file)
                    example_segment["audio"] = out_file
                    example_segment = fetch_task_media(example_segment, input_key="audio")
                    del example_segment['text']
                yield set_hashes(example_segment, overwrite=True)
        else:
            ex['transcript'] = result['text']
            ex['orig_transcript'] = result['text']
            yield ex

@recipe(
    "whisper.audio.transcribe",
    # fmt: off
    dataset=Arg(help="Dataset to write annotations into"),
    source=Arg(help="Source file to annotate"),
    model=Arg("--model", "-m", help="Name of OpenAI whisper model."),
    loader=Arg("--loader", "-lo", help="Loader to use"),
    segment=Arg("--segment", "-s", help="Segment the audio into shorter clips when annotating"),
    playpause_key=Arg("--playpause-key", "-pk", help="Keyboard shortcuts to toggle play/pause to prevent conflict with text input"),
    text_rows=Arg("--text-rows", "-tr", help="Height of text field in rows"),
    field_id=Arg("--field-id", "-fi", help="Add the transcript text to the data using this key"),
    keep_base64=Arg("--keep-base64", "-B", help="If 'audio' loader is used: don't remove base64-encoded data from the data on save"),
    autoplay=Arg("--autoplay", "-A", help="Autoplay audio when a new task loads"),
    fetch_media=Arg("--fetch-media", "-FM", help="Convert URLs and local paths to data URIs"),
    exclude=Arg("--exclude", "-e", help="Comma-separated list of dataset IDs whose annotations to exclude")
    # fmt: on
)
def whisper_audio_transcribe(
    dataset: str,
    source: SourceType,
    loader: Optional[str] = "audio",
    model: str = "base",
    segment: bool = False,
    playpause_key: List[str] = ["command+enter", "option+enter", "ctrl+enter"],
    text_rows: int = 6,
    field_id: str = "transcript",
    autoplay: bool = False,
    keep_base64: bool = False,
    fetch_media: bool = False,
    exclude: List[str] = [],
) -> ControllerComponentsDict:
    """Transcribe audio using Whisper."""
    # Available models can be found here: https://github.com/openai/whisper
    log("RECIPE: Starting recipe whisper.audio.transcribe", locals())
    stream = get_stream(source, loader=loader, rehash=True, dedup=True, is_binary=False)
    if fetch_media:
        stream.apply(fetch_media_preprocessor, input_keys=["audio"])

    def add_transcript_input_metadata(stream: StreamType) -> StreamType:
        for task in stream:
            task.update(
                {
                    "field_rows": text_rows,
                    "field_label": "Transcript",
                    "field_id": field_id,
                    "field_autofocus": True,
                }
            )
            yield task

    stream.apply(add_transcript_input_metadata)
    
    # Check if we need to tell the user that we're downloading a model
    downloaded_models = [p.stem for p in Path("~/.cache/whisper/").expanduser().glob("*.pt")]
    if model not in downloaded_models:
        print("Downloading model.")
    
    # Load model to add transcriptions
    loaded_model = whisper.load_model(model)
    stream.apply(add_annotations, model=loaded_model, model_name=model, segment=segment)
    blocks = [{"view_id": "audio"}, {"view_id": "text_input"}]

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "before_db": remove_base64 if not keep_base64 else None,
        "exclude": exclude,
        "config": {
            "blocks": blocks,
            "audio_autoplay": autoplay,
            "keymap": {"playpause": playpause_key},
            "auto_count_stream": True,
            "show_audio_timeline": segment,
            "show_audio_minimap": not segment,
            "audio_bar_width": 1,
        },
    }


def remove_base64(examples: List[TaskType]) -> List[TaskType]:
    """Remove base64-encoded string if "path" is preserved in example."""
    for eg in examples:
        if "audio" in eg and eg["audio"].startswith("data:") and "path" in eg:
            del eg["audio"]
    return examples


