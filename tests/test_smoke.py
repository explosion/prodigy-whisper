from prodigy_whisper import whisper_audio_transcribe


def test_smoke_no_segment(tmpdir):
    # Make sure we can train without errors
    components = whisper_audio_transcribe("xxx", "audio", model="tiny.en")
    assert "transcript" in next(components["stream"])


def test_smoke_segment(tmpdir):
    # Make sure we can train without errors
    components = whisper_audio_transcribe("xxx", "audio", model="tiny.en", segment=True)
    for ex in components["stream"]:
        assert "transcript" in ex
        assert ex["meta"]["path"] == "audio/talking.mp3"
        assert isinstance(ex["meta"]["segment_id"], int)
