from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def task_breakdown(task:str):
    """Breaks down a task into smaller programmable steps using Google GenAI."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    task_breakdown_file = os.path.join('prompts', "task_breakdown.txt")
    with open(task_breakdown_file, 'r') as f:
        task_breakdown_prompt = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[task,task_breakdown_prompt],
    )
    
    with open("breaked_task.txt", "w") as f:
        f.write(response.text)

    return response.text

@app.get("/")
async def root():
    """
    Root endpoint to check if the server is running.
    
    Returns:
        Dict[str, str]: A simple message indicating the server is running.    """
    # print("DEBUG: Root endpoint called")
    return {"message": "Server is running. Use POST /api to ask questions."}

@app.post("/api")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file and process it.

    Args:
        file (UploadFile): The file to upload.

    Returns:
        Dict[str, str]: A message indicating the result of the upload.
    """
    try:
        content = await file.read()
        text = content.decode("utf-8")  # assuming it's a text file
        breakdown = task_breakdown(text)
        print(breakdown)
        return {"filename": file.filename, "content": text}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    print("DEBUG: Starting FastAPI server on port 8000...")
    # Run the FastAPI app
    uvicorn.run(app, port=8000)