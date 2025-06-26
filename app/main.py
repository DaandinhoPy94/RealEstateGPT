import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.chains import get_chain
from pydantic import BaseModel

# Basis logging configuratie
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
chat_history = []
chain = get_chain()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: ChatRequest):
    global chat_history, chain
    if not chain:
        return JSONResponse(status_code=500, content={"error": "Chain not loaded"})
    
    try:
        logger.info(f"Received question: {req.question}")
        
        # Gebruik de moderne .invoke() methode
        result = chain.invoke({
            "question": req.question,
            "chat_history": chat_history
        })
        
        # --- DEBUGGING INFORMATIE ---
        # Print de bron-documenten die de retriever heeft gevonden
        print("\n--- DEBUG INFO ---")
        if 'source_documents' in result and result['source_documents']:
            print("Gevonden bron-documenten:")
            for doc in result['source_documents']:
                print(f"  - {doc.page_content}")
        else:
            print("Geen relevante bron-documenten gevonden.")
        print("------------------\n")
        # --- EINDE DEBUGGING ---

        answer = result.get("answer", "Geen antwoord gevonden.")
        chat_history.append((req.question, answer))
        
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Error during chat processing: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    return {"message": "RealEstateGPT API is running"}