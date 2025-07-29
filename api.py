from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from tts_cli import synthesize

app = FastAPI()

@app.get("/status", response_class=PlainTextResponse)
def status():
    return "Up and running."

@app.get("/synthesize")
def synthesize_api(
    text: str = Query(...),
    model_id: int = Query(...),
    language_id: int = Query(...),
    speaker_id: int = Query(...)
):
    try:
        buffer = synthesize(text,
                            model_id,
                            language_id=language_id,
                            speaker_id=speaker_id)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
