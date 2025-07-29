from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse
from tts_cli import synthesize

app = FastAPI()

roman_orthography = 1
syllabics_model = 2

ojibwe_bl = 2
maliseet = 0
mikmaw = 1
# cree_ro = 1
cree_syll = 3

jason_jones = 2
nancy_jones = 3
alan_tremblay = 0
mary_ginnish = 1
norm_wesley = 3

def process_codes(language_code, dialect_code, speaker_code):

    # Ojibwe
    if language_code == "oji":
        if dialect_code == "bl":
            if speaker_code == "jason-jones":
                return (roman_orthography, ojibwe_bl, jason_jones)
            if speaker_code == "nancy-jones":
                return (roman_orthography, ojibwe_bl, nancy_jones)
    
    # Maliseet
    if language_code == "mls":
        if dialect_code == "ml":
            if speaker_code == "alan-tremblay":
                return (roman_orthography, maliseet, alan_tremblay)

    # Mikmaw
    if language_code == "mkq":
        if dialect_code == "mk":
            if speaker_code == "mary-ginnish":
                return (roman_orthography, mikmaw, mary_ginnish)

    # Cree
    if language_code == "cre":
        if dialect_code == "rm":
            if speaker_code == "mary-ginnish":
                return (roman_orthography, mikmaw, mary_ginnish)
            if speaker_code == "norm-wesley":
                return (syllabics_model, cree_syll, norm_wesley)
            
    return None
    
@app.get("/status", response_class=PlainTextResponse)
def status():
    return "Up and running."

@app.get("/synthesize")
def synthesize_api(
    text: str = Query(...),
    # Language Code
    l: str = Query(...),
    # Dialect Code
    d: str = Query(...),
    # Speaker Code
    s: str = Query(...)
):
    ids = process_codes(l, d, s)

    if ids is None:
        raise HTTPException(status_code=400, detail="No model exists for requested combination of language, dialect, and speaker codes.")

    model_id, language_id, speaker_id = ids

    try:
        buffer = synthesize(text,
                            model_id,
                            language_id=language_id,
                            speaker_id=speaker_id)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
