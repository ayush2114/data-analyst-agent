# ============================================================================
# ╔═╗┌┬┐┌─┐┬─┐┌┬┐  ╔╦╗┌─┐┌┬┐┌─┐  ╔═╗┌┐┌┌─┐┬ ┬ ┬┌─┐┌┬┐  ╔═╗┌─┐┌─┐┌┐┌┌┬┐
# ╚═╗ │ ├─┤├┬┘ │    ║║├─┤ │ ├─┤  ╠═╣│││├─┤│ └┬┘└─┐ │   ╠═╣│ ┬├┤ │││ │ 
# ╚═╝ ┴ ┴ ┴┴└─ ┴   ═╩╝┴ ┴ ┴ ┴ ┴  ╩ ╩┘└┘┴ ┴┴─┘┴ └─┘ ┴   ╩ ╩└─┘└─┘┘└┘ ┴ 
# ============================================================================
#
# A FastAPI application that provides a data analyst agent service with:
#
# - Web data scraping capabilities
# - Local data analysis on uploaded files
# - LLM-powered autonomous data analysis
# - Robust model fallback and error handling
# - Support for visualizations and multiple data formats
#
# The agent can analyze data, produce insights, and generate
# visualizations in response to natural language questions.
# ============================================================================

# ===========================
# STANDARD LIBRARY IMPORTS
# ===========================
import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import logging
import subprocess
import time
import socket
import platform
import shutil
import asyncio
import traceback
from io import BytesIO, StringIO
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# ===========================
# THIRD-PARTY IMPORTS
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import httpx
import importlib.metadata
import psutil

# ===========================
# FASTAPI COMPONENTS
# ===========================
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response

# ===========================
# ENVIRONMENT & CONFIGURATION
# ===========================
from dotenv import load_dotenv

# ===========================
# IMAGE PROCESSING
# ===========================
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ===========================
# LANGCHAIN / LLM COMPONENTS
# ===========================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ===========================
# APPLICATION INITIALIZATION
# ===========================
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ========================================================
# ROBUST GEMINI LLM WITH FALLBACK IMPLEMENTATION
# ========================================================
# Configuration for multiple API keys and model fallback strategy
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

MODEL_HIERARCHY = [
    "gemini-2.5-pro",        # Most capable, first choice
    "gemini-2.5-flash",      # Good balance of capabilities
    "gemini-2.5-flash-lite", # Faster, less capable
    "gemini-2.0-flash",      # Fallback to previous generation
    "gemini-2.0-flash-lite"  # Last resort
]

MAX_RETRIES_PER_KEY = 2
TIMEOUT = 30
QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "403", "too many requests"]

if not GEMINI_KEYS:
    raise RuntimeError("No Gemini API keys found. Please set them in your environment.")

# ========================================================
# LLM WRAPPER WITH FALLBACK CAPABILITY
# ========================================================
class LLMWithFallback:
    """
    Wrapper for Gemini LLM that provides automatic fallback to alternative models 
    and API keys when encountering errors.
    
    Features:
    - Tries multiple API keys when quotas are exceeded
    - Falls back to less capable but available models
    - Tracks errors and failure rates for keys
    - Provides standard interface compatible with LangChain
    """
    def __init__(self, keys=None, models=None, temperature=0):
        self.keys = keys or GEMINI_KEYS
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature
        self.slow_keys_log = defaultdict(list)
        self.failing_keys_log = defaultdict(int)
        self.current_llm = None  # placeholder for actual ChatGoogleGenerativeAI instance

    def _get_llm_instance(self):
        """
        Attempt to create a working LLM instance by trying different models and API keys.
        Returns first successful instance or raises RuntimeError if all fail.
        """
        last_error = None
        for model in self.models:
            for key in self.keys:
                try:
                    llm_instance = ChatGoogleGenerativeAI(
                        model=model,
                        temperature=self.temperature,
                        google_api_key=key
                    )
                    self.current_llm = llm_instance
                    return llm_instance
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    if any(qk in msg for qk in QUOTA_KEYWORDS):
                        self.slow_keys_log[key].append(model)
                    self.failing_keys_log[key] += 1
                    time.sleep(0.5)
        raise RuntimeError(f"All models/keys failed. Last error: {last_error}")

    # Required by LangChain agent
    def bind_tools(self, tools):
        """Bind tools to the LLM for agent use (required by LangChain)"""
        llm_instance = self._get_llm_instance()
        return llm_instance.bind_tools(tools)

    # Keep .invoke interface
    def invoke(self, prompt):
        """Invoke the LLM with given prompt"""
        llm_instance = self._get_llm_instance()
        return llm_instance.invoke(prompt)


# Global timeout configuration for LLM operations
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))


# ========================================================
# APPLICATION ROUTES
# ========================================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the main HTML interface for the data analyst agent.
    
    Returns:
        HTMLResponse: The frontend interface HTML
    """
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


def parse_keys_and_types(raw_questions: str):
    """
    Parses the key/type section from the questions file.
    
    The format expected is:
    - `key_name`: type
    
    Args:
        raw_questions: String containing questions with type annotations
        
    Returns:
        tuple: (keys_list, type_map) where:
            - keys_list is an ordered list of keys
            - type_map maps each key to its corresponding casting function
    """
    import re
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float
    }
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map


# ========================================================
# DATA ACQUISITION TOOLS
# ========================================================
@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame.
    
    Supports multiple data formats:
    - HTML tables
    - CSV
    - Excel
    - Parquet
    - JSON
    - Plain text (fallback)
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dict with keys:
        - status: "success" or "error"
        - data: List of records (if successful)
        - columns: List of column names (if successful)
        - message: Error message (if error)
    """
    print(f"Scraping URL: {url}")
    try:
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        # --- CSV ---
        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))

        # --- Excel ---
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))

        # --- Parquet ---
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))

        # --- JSON ---
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])

        # --- HTML / Fallback ---
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            # Try HTML tables first
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        # --- Unknown type fallback ---
        else:
            df = pd.DataFrame({"text": [resp.text]})

        # --- Normalize columns ---
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ========================================================
# UTILITIES FOR SAFE CODE EXECUTION
# ========================================================
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    
    Handles various JSON formatting scenarios and cleans the output.
    
    Args:
        output: Raw text output from LLM
        
    Returns:
        Dict: Parsed JSON object or error details
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        # remove triple-fence markers if present
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        # find outermost JSON object by scanning for balanced braces
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(last, first, -1):
                cand = s[first:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

# Scraping function definition for user code sandboxes
SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "data": [],
            "columns": []
        }

    soup = BeautifulSoup(response.text, "html.parser")
    tables = pd.read_html(response.text)

    if tables:
        df = tables[0]  # Take first table
        df.columns = [str(c).strip() for c in df.columns]
        
        # Ensure all columns are unique and string
        df.columns = [str(col) for col in df.columns]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        # Fallback to plain text
        text_data = soup.get_text(separator="\n", strip=True)

        # Try to detect possible "keys" from text like Runtime, Genre, etc.
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])  # start empty
        for col in detected_cols:
            df[col] = None

        if df.empty:
            df["text"] = [text_data]

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
'''


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write and execute a temporary Python script in a safe environment.
    
    The script includes:
    - Standard data science imports (pandas, numpy, matplotlib)
    - Dataframe loading from pickle if provided
    - Helper functions for image conversion and scraping
    - The user's code, which populates a 'results' dictionary
    
    Args:
        code: Python code to execute
        injected_pickle: Path to pickled DataFrame (optional)
        timeout: Maximum execution time in seconds
        
    Returns:
        Dict: Execution results or error information
    """
    # create file content
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
    # inject df if a pickle path provided
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        # ensure data exists so user code that references data won't break
        preamble.append("data = globals().get('data', {})\n")

    # plot_to_base64 helper that tries to reduce size under 100_000 bytes
    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    # try decreasing dpi/figure size iteratively
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    # if Pillow available, try convert to WEBP which is typically smaller
    try:
        from PIL import Image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=40)
        buf.seek(0)
        im = Image.open(buf)
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=80, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
        # try lower quality
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    # as last resort return downsized PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    # Build the code to write
    script_lines = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    # ensure results printed as json
    script_lines.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run([sys.executable, tmp_path],
                                   capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            # collect stderr and stdout for debugging
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        # parse stdout as json
        out = completed.stdout.strip()
        try:
            parsed = json.loads(out)
            return parsed
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass


# ========================================================
# LLM AGENT CONFIGURATION
# ========================================================
# Initialize LLM with fallback capabilities
llm = LLMWithFallback(temperature=0)

# Tools list for agent (LangChain tool decorator returns metadata for the LLM)
tools = [scrape_url_to_dataframe]  # we only expose scraping as a tool; agent will still produce code

# System prompt that instructs the agent on how to structure responses
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
5. When returning plots, always use `plot_to_base64()` to keep image sizes small.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent with tool-calling capabilities
agent = create_tool_calling_agent(
    llm=llm,
    tools=[scrape_url_to_dataframe],  # let the agent call tools if it wants; we will also pre-process scrapes
    prompt=prompt
)

# Set up agent executor to manage agent interactions
agent_executor = AgentExecutor(
    agent=agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)


# ========================================================
# AGENT EXECUTION ENGINE
# ========================================================
def run_agent_safely(llm_input: str) -> Dict:
    """
    Orchestrates the complete agent workflow:
    
    1. Gets LLM response to the input
    2. Extracts code and questions from JSON
    3. Pre-processes any scraping requests
    4. Executes the code safely
    5. Maps question strings to answers
    
    Args:
        llm_input: Formatted input with rules, questions and data preview
        
    Returns:
        Dict: Results mapping questions to answers, or error information
    """
    try:
        response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
        if not raw_out:
            return {"error": f"Agent returned no output. Full response: {response}"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if not isinstance(parsed, dict) or "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response format: {parsed}"}

        code = parsed["code"]
        questions: List[str] = parsed["questions"]

        # Detect scrape calls; find all URLs used in scrape_url_to_dataframe("URL")
        urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
        pickle_path = None
        if urls:
            # For now support only the first URL (agent may code multiple scrapes; you can extend this)
            url = urls[0]
            tool_resp = scrape_url_to_dataframe(url)
            if tool_resp.get("status") != "success":
                return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
            # create df and pickle it
            df = pd.DataFrame(tool_resp["data"])
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name
            # Make sure agent's code can reference df/data: we will inject the pickle loader in the temp script

        # Execute code in temp python script
        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message', exec_result)}", "raw": exec_result.get("raw")}

        # exec_result['result'] should be results dict
        results_dict = exec_result.get("result", {})
        # Map to original questions (they asked to use exact question strings)
        output = {}
        for q in questions:
            output[q] = results_dict.get(q, "Answer not found")
        return output

    except Exception as e:
        logger.exception("run_agent_safely failed")
        return {"error": str(e)}


# ========================================================
# API ENDPOINTS
# ========================================================
@app.post("/api")
async def analyze_data(request: Request):
    """
    Main endpoint for data analysis.
    
    Accepts:
    - questions_file: Text file with questions to answer
    - data_file: (Optional) Dataset to analyze
    
    Process:
    1. Parses questions and type hints
    2. Loads and processes any uploaded data
    3. Runs the LLM agent to generate and execute analysis code
    4. Returns structured results
    
    Returns:
        JSONResponse: Results of the analysis or error details
    """
    try:
        form = await request.form()
        questions_file = None
        data_file = None

        for key, val in form.items():
            if hasattr(val, "filename") and val.filename:  # it's a file
                fname = val.filename.lower()
                if fname.endswith(".txt") and questions_file is None:
                    questions_file = val
                else:
                    data_file = val

        if not questions_file:
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")
        keys_list, type_map = parse_keys_and_types(raw_questions)

        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        if data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            from io import BytesIO

            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError:
                    df = pd.DataFrame(json.loads(content.decode("utf-8")))
            elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                try:
                    if PIL_AVAILABLE:
                        image = Image.open(BytesIO(content))
                        image = image.convert("RGB")  # ensure RGB format
                        df = pd.DataFrame({"image": [image]})
                    else:
                        raise HTTPException(400, "PIL not available for image processing")
                except Exception as e:
                    raise HTTPException(400, f"Image processing failed: {str(e)}")  
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            # Pickle for injection
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        # Build rules based on data presence
        if dataset_uploaded:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the uploaded dataset for answering questions.\n"
                "4) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "5) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
                "2) Produce a final JSON object with keys:\n"
                '   - "questions": [ ... original question strings ... ]\n'
                '   - "code": "..."  (Python code that fills `results` with exact question strings as keys)\n'
                "3) For plots: use plot_to_base64() helper to return base64 image data under 100kB.\n"
            )

        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview if df_preview else ''}"
            "Respond with the JSON object only."
        )

        # Run agent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path)
            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # Post-process key mapping & type casting
        if keys_list and type_map:
            mapped = {}
            for idx, q in enumerate(result.keys()):
                if idx < len(keys_list):
                    key = keys_list[idx]
                    caster = type_map.get(key, str)
                    try:
                        val = result[q]
                        if isinstance(val, str) and val.startswith("data:image/"):
                            # Remove data URI prefix
                            val = val.split(",", 1)[1] if "," in val else val
                        mapped[key] = caster(val) if val not in (None, "") else val
                    except Exception:
                        mapped[key] = result[q]
            result = mapped

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Unified agent execution with retries and error handling.
    
    This function:
    1. Retries LLM calls if needed
    2. Processes scraped data or uses provided data
    3. Executes code safely and returns results
    
    Args:
        llm_input: The formatted input for the LLM agent
        pickle_path: Optional path to pickled DataFrame
        
    Returns:
        Dict: Results mapping questions to answers or error details
    """
    try:
        max_retries = 3
        raw_out = ""
        for attempt in range(1, max_retries + 1):
            response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
            raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
            if raw_out:
                break
        if not raw_out:
            return {"error": f"Agent returned no output after {max_retries} attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response: {parsed}"}

        code = parsed["code"]
        questions = parsed["questions"]

        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                df = pd.DataFrame(tool_resp["data"])
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


# ========================================================
# UTILITY ROUTES
# ========================================================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Serve favicon.ico if present in the working directory.
    Otherwise return a tiny transparent PNG to avoid 404s.
    """
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")


@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    """Health/info endpoint. Use POST /api for actual analysis."""
    return JSONResponse({
        "ok": True,
        "message": "✅ Server is live. Send a POST request to /api with 'questions_file' (required) and 'data_file' (optional).",
    })


# ========================================================
# SYSTEM DIAGNOSTICS MODULE
# ========================================================

# Configuration for diagnostics
DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "GitHub": "https://api.github.com",
}
DIAG_LLM_KEY_TIMEOUT = 30  # seconds per key/model simple ping test
DIAG_PARALLELISM = 6       # thread workers for sync checks
RUN_LONGER_CHECKS = False  # Playwright/duckdb tests run only if true

# Use existing keys/models or create empty lists
try:
    _GEMINI_KEYS = GEMINI_KEYS
    _MODEL_HIERARCHY = MODEL_HIERARCHY
except NameError:
    _GEMINI_KEYS = []
    _MODEL_HIERARCHY = []

# Create 1×1 transparent PNG fallback for favicon
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

# Helper: ISO timestamp generator
def _now_iso():
    """Return current UTC time in ISO 8601 format with Z suffix"""
    return datetime.utcnow().isoformat() + "Z"

# Helper: run sync function in threadpool with timeout
_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)
async def run_in_thread(fn, *a, timeout=30, **kw):
    """
    Run a synchronous function in a threadpool with timeout.
    Propagates exceptions for better error handling.
    """
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("timeout")
    except Exception as e:
        # re-raise for caller to capture stacktrace easily
        raise


# ---- Diagnostic check functions ----

def _env_check(required=None):
    """Check for presence of required environment variables"""
    required = required or []
    out = {}
    for k in required:
        out[k] = {"present": bool(os.getenv(k)), "masked": (os.getenv(k)[:4] + "..." + os.getenv(k)[-4:]) if os.getenv(k) else None}
    # Also include simple helpful values
    out["GOOGLE_MODEL"] = os.getenv("GOOGLE_MODEL")
    out["LLM_TIMEOUT_SECONDS"] = os.getenv("LLM_TIMEOUT_SECONDS")
    return out

def _system_info():
    """Collect system information for diagnostics"""
    info = {
        "host": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
    }
    # disk free for app dir and tmp
    try:
        _cwd = os.getcwd()
        info["cwd_free_gb"] = round(shutil.disk_usage(_cwd).free / 1024**3, 2)
    except Exception:
        info["cwd_free_gb"] = None
    try:
        info["tmp_free_gb"] = round(shutil.disk_usage(tempfile.gettempdir()).free / 1024**3, 2)
    except Exception:
        info["tmp_free_gb"] = None
    # GPU quick probe (if torch installed)
    try:
        import torch
        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_installed"] = False
        info["cuda_available"] = False
    return info

def _temp_write_test():
    """Test write access to temp directory"""
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"tmp_dir": tmp, "write_ok": ok}

def _app_write_test():
    """Test write access to application directory"""
    cwd = os.getcwd()
    path = os.path.join(cwd, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as f:
        f.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"cwd": cwd, "write_ok": ok}

def _pandas_pipeline_test():
    """Test basic pandas functionality"""
    import pandas as _pd
    df = _pd.DataFrame({"x":[1,2,3], "y":[4,5,6]})
    df["z"] = df["x"] * df["y"]
    agg = df["z"].sum()
    return {"rows": df.shape[0], "cols": df.shape[1], "z_sum": int(agg)}

def _installed_packages_sample():
    """Get sample of installed Python packages"""
    try:
        out = []
        for dist in importlib.metadata.distributions():
            try:
                out.append(f"{dist.metadata['Name']}=={dist.version}")
            except Exception:
                try:
                    out.append(f"{dist.metadata['Name']}")
                except Exception:
                    continue
        return {"sample_packages": sorted(out)[:20]}
    except Exception as e:
        return {"error": str(e)}

def _network_probe_sync(url, timeout=30):
    """Test network connectivity to a specific URL"""
    try:
        r = requests.head(url, timeout=timeout)
        return {"ok": True, "status_code": r.status_code, "latency_ms": int(r.elapsed.total_seconds()*1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _test_gemini_key_model(key, model, ping_text="ping"):
    """
    Test a Gemini API key by sending a minimal request.
    Returns structured info about the test results.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        return {"ok": False, "error": f"langchain_google_genai import error: {e}"}

    try:
        obj = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=key
        )

        def extract_text(resp):
            """Normalize any type of LLM response into a clean string."""
            try:
                if resp is None:
                    return None
                if isinstance(resp, str):
                    return resp
                if hasattr(resp, "content") and isinstance(resp.content, str):
                    return resp.content
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    return resp.text
                # For objects with .dict() method
                if hasattr(resp, "dict"):
                    try:
                        return str(resp.dict())
                    except Exception:
                        pass
                return str(resp)
            except Exception as e:
                return f"[unreadable response: {e}]"

        # First try invoke()
        try:
            resp = obj.invoke(ping_text)
            text = extract_text(resp)
            return {"ok": True, "model": model, "summary": text[:200] if text else None}
        except Exception as e_invoke:
            # Try __call__()
            try:
                resp = obj.__call__(ping_text)
                text = extract_text(resp)
                return {"ok": True, "model": model, "summary": text[:200] if text else None}
            except Exception as e_call:
                return {"ok": False, "error": f"invoke failed: {e_invoke}; call failed: {e_call}"}

    except Exception as e_outer:
        return {"ok": False, "error": str(e_outer)}


# ---- Async network diagnostic functions ----

async def check_network():
    """
    Check connectivity to multiple important API endpoints in parallel.
    Returns status and latency information for each endpoint.
    """
    coros = []
    for name, url in DIAG_NETWORK_TARGETS.items():
        coros.append(run_in_thread(_network_probe_sync, url, timeout=30))
    results = await asyncio.gather(*[asyncio.create_task(c) for c in coros], return_exceptions=True)
    out = {}
    for (name, _), res in zip(DIAG_NETWORK_TARGETS.items(), results):
        if isinstance(res, Exception):
            out[name] = {"ok": False, "error": str(res)}
        else:
            out[name] = res
    return out

async def check_llm_keys_models():
    """
    Test all Gemini API keys against each model in priority order.
    Stops after finding first working key/model combination.
    """
    if not _GEMINI_KEYS:
        return {"warning": "no GEMINI_KEYS configured"}

    results = []
    # we will stop early if we find a working combo but still record attempts
    for model in (_MODEL_HIERARCHY or ["gemini-2.5-pro"]):
        # test keys in parallel for this model
        tasks = []
        for key in _GEMINI_KEYS:
            tasks.append(run_in_thread(_test_gemini_key_model, key, model, timeout=DIAG_LLM_KEY_TIMEOUT))
        completed = await asyncio.gather(*[asyncio.create_task(t) for t in tasks], return_exceptions=True)
        model_summary = {"model": model, "attempts": []}
        any_ok = False
        for key, res in zip(_GEMINI_KEYS, completed):
            if isinstance(res, Exception):
                model_summary["attempts"].append({"key_mask": (key[:4] + "..." + key[-4:]) if key else None, "ok": False, "error": str(res)})
            else:
                # res is dict returned by _test_gemini_key_model
                model_summary["attempts"].append({"key_mask": (key[:4] + "..." + key[-4:]) if key else None, **res})
                if res.get("ok"):
                    any_ok = True
        results.append(model_summary)
        if any_ok:
            # stop once first model has a working key (respecting MODEL_HIERARCHY)
            break
    return {"models_tested": results}

# ---- Optional extended diagnostic checks ----

async def check_duckdb():
    """Test if DuckDB is available and functional"""
    try:
        import duckdb
        def duck_check():
            conn = duckdb.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            return {"duckdb": True}
        return await run_in_thread(duck_check, timeout=30)
    except Exception as e:
        return {"duckdb_error": str(e)}

async def check_playwright():
    """Test if Playwright browser automation is available and functional"""
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            b = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await b.new_page()
            await page.goto("about:blank")
            ua = await page.evaluate("() => navigator.userAgent")
            await b.close()
            return {"playwright_ok": True, "ua": ua[:200]}
    except Exception as e:
        return {"playwright_error": str(e)}

# ---- System diagnostics API endpoint ----

@app.get("/summary")
async def diagnose(full: bool = Query(False, description="If true, run extended checks (duckdb/playwright)")):
    """
    Comprehensive system diagnostics endpoint.
    
    Performs multiple checks in parallel:
    - Environment variables
    - System information
    - File system access
    - Pandas functionality
    - Network connectivity
    - LLM API key testing
    - Optional: DuckDB and Playwright testing
    
    Args:
        full: Whether to run extended checks
        
    Returns:
        Dict: Complete diagnostic report with status of all checks
    """
    started = datetime.utcnow()
    report = {
        "status": "ok",
        "server_time": _now_iso(),
        "summary": {},
        "checks": {},
        "elapsed_seconds": None
    }

    # prepare tasks
    tasks = {
        "env": run_in_thread(_env_check, ["GOOGLE_API_KEY", "GOOGLE_MODEL", "LLM_TIMEOUT_SECONDS"], timeout=3),
        "system": run_in_thread(_system_info, timeout=30),
        "tmp_write": run_in_thread(_temp_write_test, timeout=30),
        "cwd_write": run_in_thread(_app_write_test, timeout=30),
        "pandas": run_in_thread(_pandas_pipeline_test, timeout=30),
        "packages": run_in_thread(_installed_packages_sample, timeout=50),
        "network": asyncio.create_task(check_network()),
        "llm_keys_models": asyncio.create_task(check_llm_keys_models())
    }

    if full or RUN_LONGER_CHECKS:
        tasks["duckdb"] = asyncio.create_task(check_duckdb())
        tasks["playwright"] = asyncio.create_task(check_playwright())

    # run all concurrently, collect results
    results = {}
    for name, coro in tasks.items():
        try:
            res = await coro
            results[name] = {"status": "ok", "result": res}
        except TimeoutError:
            results[name] = {"status": "timeout", "error": "check timed out"}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e), "trace": traceback.format_exc()}

    report["checks"] = results

    # quick summary flags
    failed = [k for k, v in results.items() if v.get("status") != "ok"]
    if failed:
        report["status"] = "warning"
        report["summary"]["failed_checks"] = failed
    else:
        report["status"] = "ok"
        report["summary"]["failed_checks"] = []

    report["elapsed_seconds"] = (datetime.utcnow() - started).total_seconds()
    return report


# ========================================================
# APPLICATION ENTRY POINT
# ========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))