from fastapi import FastAPI
from gradio_client import Client

API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"

# set up the Gradio client
client = Client(API_URL)

def transcribe_audio(audio_path, task="transcribe", return_timestamps=False):
    """Function to transcribe an audio file using the Whisper JAX endpoint."""
    if task not in ["transcribe", "translate"]:
        raise ValueError("task should be one of 'transcribe' or 'translate'.")

    text, runtime = client.predict(
        audio_path,
        task,
        return_timestamps,
        api_name="/predict_1",
    )
    return text



app = FastAPI()

@app.get("/")
def read_root():
    return { "hello": "world" }

@app.get("/transcribe/")
def transcribe_file(file_path: str):
    # transcribe audio
    # e.g. `curl "http://localhost:8000/transcribe/?file_path=https://d1gurz8aso75n6.cloudfront.net/original/3X/8/7/870b394e46448256a12afaa4619bd74c974d31ba.m4a"`
    output = transcribe_audio(file_path)

    return {"transcription": output}