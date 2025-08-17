# =============================
# Standard Library Imports
# =============================
import asyncio
import base64
import concurrent.futures
import gc
import importlib.metadata
import io
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime
from functools import partial

# =============================
# Third-Party Imports
# =============================
import numpy as np
import pandas as pd
import psutil
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional plotting libs (used inside sandbox runner)
import matplotlib.pyplot as plt  # noqa: F401 (imported to mirror original environment)

# Optional Pillow import
try:
    from PIL import Image  # type: ignore

    PIL_AVAILABLE = True
except Exception:  # pragma: no cover - environment-specific
    PIL_AVAILABLE = False

# =============================
# App Setup & Logging
# =============================
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tds-agent")

app = FastAPI(title="TDS Data Analyst Agent")

# =============================
# LLM Configuration & Fallback
# =============================
GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

MODEL_HIERARCHY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

MAX_RETRIES_PER_KEY = 2  # retained for parity; not directly used below
TIMEOUT = 30  # retained for parity; not directly used below
QUOTA_KEYWORDS = ["quota", "exceeded", "rate limit", "403", "too many requests"]

if not GEMINI_KEYS:
    raise RuntimeError("No Gemini API keys found. Please set them in your environment.")


class LLMWithFallback:
    """Thin wrapper that tries model/key combos until one initializes.

    It exposes `bind_tools` and `invoke` to match LangChain expectations.
    """

    def __init__(self, keys=None, models=None, temperature: float = 0):
        self.keys = keys or GEMINI_KEYS
        self.models = models or MODEL_HIERARCHY
        self.temperature = temperature
        self.slow_keys_log = defaultdict(list)
        self.failing_keys_log = defaultdict(int)
        self.current_llm = None

    def _activate_llm(self) -> ChatGoogleGenerativeAI:
        last_exc: Exception | None = None
        for model in self.models:
            for key in self.keys:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=model, temperature=self.temperature, google_api_key=key
                    )
                    self.current_llm = llm
                    return llm
                except Exception as e:  # pragma: no cover - network/env dependent
                    last_exc = e
                    msg = str(e).lower()
                    if any(t in msg for t in QUOTA_KEYWORDS):
                        self.slow_keys_log[key].append(model)
                    self.failing_keys_log[key] += 1
                    time.sleep(0.5)
        raise RuntimeError(f"All models/keys failed. Last error: {last_exc}")

    # LangChain compatibility
    def bind_tools(self, tools):
        return self._activate_llm().bind_tools(tools)

    def invoke(self, prompt):
        return self._activate_llm().invoke(prompt)


LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 240))


# =============================
# Static Frontend
# =============================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Return index.html if available; otherwise minimal guidance page."""
    try:
        with open("index.html", "r", encoding="utf-8") as fh:
            return HTMLResponse(content=fh.read())
    except FileNotFoundError:
        return HTMLResponse(
            content=(
                "<h1>Frontend not found</h1>"
                "<p>Please ensure index.html is in the same directory as app.py</p>"
            ),
            status_code=404,
        )


# =============================
# Utility: parse key/type spec from questions file
# =============================

def parse_keys_and_types(raw_questions: str):
    """
    Parse a section of backticked keys with declared types and build a cast map.

    Returns:
        keys_list: ordered list of keys
        type_map: mapping of key -> callable caster
    """
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    pairs = re.findall(pattern, raw_questions)
    cast_table = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float,
    }
    type_map = {k: cast_table.get(t.lower(), str) for k, t in pairs}
    ordered_keys = [k for k, _ in pairs]
    return ordered_keys, type_map


# =============================
# LangChain Tool: scrape_url_to_dataframe
# =============================

@tool
def scrape_url_to_dataframe(url: str) -> dict:
    """Fetch a URL and convert its contents into a DataFrame-like payload.

    Supports CSV, Excel, Parquet, JSON, and HTML (tables preferred, text fallback).
    Always returns {"status": "success", "data": [...], "columns": [...]} on success.
    """
    print(f"Scraping URL: {url}")
    try:
        from bs4 import BeautifulSoup
        from io import BytesIO, StringIO

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
        ctype = (resp.headers.get("Content-Type", "").lower())

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
                payload = resp.json()
                df = pd.json_normalize(payload)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])

        # --- HTML / Fallback ---
        elif "text/html" in ctype or re.search(r"/wiki/|\.org|\.com", url, re.IGNORECASE):
            html = resp.text
            try:
                tables = pd.read_html(StringIO(html), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass

            if df is None:
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        else:  # unknown content-type
            df = pd.DataFrame({"text": [resp.text]})

        # Normalize column names
        df.columns = (
            df.columns.map(str).str.replace(r"\[.*\]", "", regex=True).str.strip()
        )

        return {"status": "success", "data": df.to_dict("records"), "columns": df.columns.tolist()}

    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================
# LLM Output Cleaning
# =============================

def clean_llm_output(output: str) -> dict:
    """Extract the outermost JSON object from an LLM string output."""
    try:
        if not output:
            return {"error": "Empty LLM output"}
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first : last + 1]
        try:
            return json.loads(candidate)
        except Exception as e:
            for i in range(last, first, -1):
                try:
                    return json.loads(s[first : i + 1])
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {e}", "raw": candidate}
    except Exception as e:  # pragma: no cover - defensive
        return {"error": str(e)}


# =============================
# Embedded scrape function text for sandbox runner
# =============================
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
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]
        df.columns = [str(col) for col in df.columns]
        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns)
        }
    else:
        text_data = soup.get_text(separator="\n", strip=True)
        detected_cols = set(re.findall(r"\b[A-Z][a-zA-Z ]{2,15}\b", text_data))
        df = pd.DataFrame([{}])
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


# =============================
# Sandbox Runner (executes agent-produced code)
# =============================

def write_and_run_temp_python(code: str, injected_pickle: str | None = None, timeout: int = 60) -> dict:
    """Create a temporary Python script, optionally inject a pickled DataFrame, run it, and parse JSON output."""
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

    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        preamble.append("data = globals().get('data', {})\n")

    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
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
        out_buf = BytesIO()
        im.save(out_buf, format='WEBP', quality=60, method=6)
        out_buf.seek(0)
        ob = out_buf.getvalue()
        if len(ob) <= max_bytes:
            return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''

    script_lines: list[str] = []
    script_lines.extend(preamble)
    script_lines.append(helper)
    script_lines.append(SCRAPE_FUNC)
    script_lines.append("\nresults = {}\n")
    script_lines.append(code)
    script_lines.append(
        "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"
    )

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
    tmp.write("\n".join(script_lines))
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()

    try:
        completed = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        out = completed.stdout.strip()
        try:
            return json.loads(out)
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {e}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:  # pragma: no cover
            pass


# =============================
# Agent & Prompt Setup
# =============================
# Use resilient LLM wrapper
llm = LLMWithFallback(temperature=0)

# Only expose the scraping tool to the agent
TOOLS = [scrape_url_to_dataframe]

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a full-stack autonomous data analyst agent.\n\n"
                "You will receive:\n"
                "- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)\n"
                "- One or more **questions**\n"
                "- An optional **dataset preview**\n\n"
                "You must:\n"
                "1. Follow the provided rules exactly.\n"
                "2. Return only a valid JSON object — no extra commentary or formatting.\n"
                "3. The JSON must contain:\n"
                "   - \"questions\": [ list of original question strings exactly as provided ]\n"
                "   - \"code\": \"...\" (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)\n"
                "4. Your Python code will run in a sandbox with:\n"
                "   - pandas, numpy, matplotlib available\n"
                "   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.\n"
                "5. When returning plots, always use `plot_to_base64()` to keep image sizes small.\n"
                "6. Make sure all variables are defined before use, and the code can run without any undefined references.\n"
            ),
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=PROMPT)

agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False,
)


# =============================
# Agent Runners
# =============================

def run_agent_safely(llm_input: str) -> dict:
    """Run agent, extract JSON, optionally pre-scrape, execute code, and map results to questions."""
    try:
        response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        raw = response.get("output") or response.get("final_output") or response.get("text") or ""
        if not raw:
            return {"error": f"Agent returned no output. Full response: {response}"}

        parsed = clean_llm_output(raw)
        if "error" in parsed:
            return parsed
        if not isinstance(parsed, dict) or "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response format: {parsed}"}

        code = parsed["code"]
        questions: list[str] = parsed["questions"]

        # Detect scrape calls up-front, run once, and pickle for injected df
        urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
        pickle_path = None
        if urls:
            url = urls[0]
            tool_resp = scrape_url_to_dataframe(url)
            if tool_resp.get("status") != "success":
                return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
            df = pd.DataFrame(tool_resp["data"])  # normalize to DF
            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message', exec_result)}", "raw": exec_result.get("raw")}

        out_map = {}
        produced = exec_result.get("result", {})
        for q in questions:
            out_map[q] = produced.get(q, "Answer not found")
        return out_map
    except Exception as e:  # pragma: no cover
        log.exception("run_agent_safely failed")
        return {"error": str(e)}


def run_agent_safely_unified(llm_input: str, pickle_path: str | None = None) -> dict:
    """Robust runner with retries; inject DF when provided else allow scrape injection."""
    try:
        max_retries = 3
        raw_out = ""
        for _ in range(max_retries):
            resp = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
            raw_out = resp.get("output") or resp.get("final_output") or resp.get("text") or ""
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
                df = pd.DataFrame(tool_resp["data"])  # normalize
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}
    except Exception as e:  # pragma: no cover
        log.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


# =============================
# API Routes
# =============================

@app.post("/api")
async def analyze_data(request: Request):
    """Main analysis route accepting a questions file and optional dataset file."""
    try:
        form = await request.form()
        questions_file: UploadFile | None = None
        data_file: UploadFile | None = None

        for _, val in form.items():
            if hasattr(val, "filename") and val.filename:
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
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                if not PIL_AVAILABLE:
                    raise HTTPException(400, "PIL not available for image processing")
                try:
                    image = Image.open(BytesIO(content))
                    image = image.convert("RGB")
                    df = pd.DataFrame({"image": [image]})
                except Exception as e:
                    raise HTTPException(400, f"Image processing failed: {e}")
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        # Build rules communicated to the agent
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

        # Execute agent with a timeout using a thread
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path)
            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # Map outputs to requested keys and cast types
        if keys_list and type_map:
            transformed: dict[str, object] = {}
            for idx, q in enumerate(result.keys()):
                if idx < len(keys_list):
                    key = keys_list[idx]
                    caster = type_map.get(key, str)
                    try:
                        val = result[q]
                        if isinstance(val, str) and val.startswith("data:image/"):
                            val = val.split(",", 1)[1] if "," in val else val
                        transformed[key] = caster(val) if val not in (None, "") else val
                    except Exception:
                        transformed[key] = result[q]
            result = transformed

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover
        log.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


# Health/info endpoint (GET)
@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    return JSONResponse(
        {
            "ok": True,
            "message": "Server is running. Use POST /api with 'questions_file' and optional 'data_file'.",
        }
    )


# =============================
# Favicon Handling
# =============================
# Tiny 1×1 transparent PNG fallback
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")


# =============================
# System Diagnostics
# =============================
import httpx  # noqa: F401 (import to mirror environment)
import shutil
from concurrent.futures import ThreadPoolExecutor

# Network probes
DIAG_NETWORK_TARGETS = {
    "Google AI": "https://generativelanguage.googleapis.com",
    "AISTUDIO": "https://aistudio.google.com/",
    "OpenAI": "https://api.openai.com",
    "GitHub": "https://api.github.com",
}
DIAG_LLM_KEY_TIMEOUT = 30
DIAG_PARALLELISM = 6
RUN_LONGER_CHECKS = False

try:
    _GEMINI_KEYS = GEMINI_KEYS
    _MODEL_HIERARCHY = MODEL_HIERARCHY
except NameError:  # pragma: no cover
    _GEMINI_KEYS = []
    _MODEL_HIERARCHY = []


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


_executor = ThreadPoolExecutor(max_workers=DIAG_PARALLELISM)


async def run_in_thread(fn, *a, timeout=30, **kw):
    loop = asyncio.get_running_loop()
    try:
        task = loop.run_in_executor(_executor, partial(fn, *a, **kw))
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("timeout")


def _env_check(required=None):
    required = required or []
    out = {}
    for k in required:
        val = os.getenv(k)
        out[k] = {"present": bool(val), "masked": (val[:4] + "..." + val[-4:]) if val else None}
    out["GOOGLE_MODEL"] = os.getenv("GOOGLE_MODEL")
    out["LLM_TIMEOUT_SECONDS"] = os.getenv("LLM_TIMEOUT_SECONDS")
    return out


def _system_info():
    info = {
        "host": socket.gethostname(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
    }
    try:
        cwd = os.getcwd()
        info["cwd_free_gb"] = round(shutil.disk_usage(cwd).free / 1024 ** 3, 2)
    except Exception:
        info["cwd_free_gb"] = None
    try:
        info["tmp_free_gb"] = round(shutil.disk_usage(tempfile.gettempdir()).free / 1024 ** 3, 2)
    except Exception:
        info["tmp_free_gb"] = None

    try:
        import torch  # type: ignore

        info["torch_installed"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_installed"] = False
        info["cuda_available"] = False
    return info


def _temp_write_test():
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as fh:
        fh.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"tmp_dir": tmpdir, "write_ok": ok}


def _app_write_test():
    cwd = os.getcwd()
    path = os.path.join(cwd, f"diag_test_{int(time.time())}.tmp")
    with open(path, "w") as fh:
        fh.write("ok")
    ok = os.path.exists(path)
    os.remove(path)
    return {"cwd": cwd, "write_ok": ok}


def _pandas_pipeline_test():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    df["z"] = df["x"] * df["y"]
    return {"rows": df.shape[0], "cols": df.shape[1], "z_sum": int(df["z"].sum())}


def _installed_packages_sample():
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
    try:
        r = requests.head(url, timeout=timeout)
        return {"ok": True, "status_code": r.status_code, "latency_ms": int(r.elapsed.total_seconds() * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _test_gemini_key_model(key, model, ping_text="ping"):
    """Light-touch request to validate a key/model pair."""
    try:
        obj = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=key)

        def extract_text(resp):
            try:
                if resp is None:
                    return None
                if isinstance(resp, str):
                    return resp
                if hasattr(resp, "content") and isinstance(resp.content, str):
                    return resp.content
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    return resp.text
                if hasattr(resp, "dict"):
                    try:
                        return str(resp.dict())
                    except Exception:
                        pass
                return str(resp)
            except Exception as e:  # pragma: no cover
                return f"[unreadable response: {e}]"

        try:
            resp = obj.invoke(ping_text)
            text = extract_text(resp)
            return {"ok": True, "model": model, "summary": text[:200] if text else None}
        except Exception as e_invoke:
            try:
                resp = obj.__call__(ping_text)  # fallback path
                text = extract_text(resp)
                return {"ok": True, "model": model, "summary": text[:200] if text else None}
            except Exception as e_call:  # pragma: no cover
                return {"ok": False, "error": f"invoke failed: {e_invoke}; call failed: {e_call}"}
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": str(e)}


async def check_network():
    tasks = [run_in_thread(_network_probe_sync, url, timeout=30) for _, url in DIAG_NETWORK_TARGETS.items()]
    results = await asyncio.gather(*[asyncio.create_task(t) for t in tasks], return_exceptions=True)
    out = {}
    for (name, _), res in zip(DIAG_NETWORK_TARGETS.items(), results):
        if isinstance(res, Exception):
            out[name] = {"ok": False, "error": str(res)}
        else:
            out[name] = res
    return out


async def check_llm_keys_models():
    if not _GEMINI_KEYS:
        return {"warning": "no GEMINI_KEYS configured"}

    results = []
    for model in (_MODEL_HIERARCHY or ["gemini-2.5-pro"]):
        tasks = [run_in_thread(_test_gemini_key_model, key, model, timeout=DIAG_LLM_KEY_TIMEOUT) for key in _GEMINI_KEYS]
        completed = await asyncio.gather(*[asyncio.create_task(t) for t in tasks], return_exceptions=True)
        model_summary = {"model": model, "attempts": []}
        any_ok = False
        for key, res in zip(_GEMINI_KEYS, completed):
            if isinstance(res, Exception):
                model_summary["attempts"].append({"key_mask": (key[:4] + "..." + key[-4:]) if key else None, "ok": False, "error": str(res)})
            else:
                model_summary["attempts"].append({"key_mask": (key[:4] + "..." + key[-4:]) if key else None, **res})
                if res.get("ok"):
                    any_ok = True
        results.append(model_summary)
        if any_ok:
            break
    return {"models_tested": results}


async def check_duckdb():  # optional
    try:
        import duckdb  # noqa: F401

        def duck_check():
            conn = duckdb.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            return {"duckdb": True}

        return await run_in_thread(duck_check, timeout=30)
    except Exception as e:  # pragma: no cover
        return {"duckdb_error": str(e)}


async def check_playwright():  # optional, can be slow
    try:
        from playwright.async_api import async_playwright  # type: ignore

        async with async_playwright() as p:
            b = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await b.new_page()
            await page.goto("about:blank")
            ua = await page.evaluate("() => navigator.userAgent")
            await b.close()
            return {"playwright_ok": True, "ua": ua[:200]}
    except Exception as e:  # pragma: no cover
        return {"playwright_error": str(e)}


@app.get("/summary")
async def diagnose(full: bool = Query(False, description="If true, run extended checks (duckdb/playwright)")):
    started = datetime.utcnow()
    report = {
        "status": "ok",
        "server_time": _now_iso(),
        "summary": {},
        "checks": {},
        "elapsed_seconds": None,
    }

    tasks = {
        "env": run_in_thread(_env_check, ["GOOGLE_API_KEY", "GOOGLE_MODEL", "LLM_TIMEOUT_SECONDS"], timeout=3),
        "system": run_in_thread(_system_info, timeout=30),
        "tmp_write": run_in_thread(_temp_write_test, timeout=30),
        "cwd_write": run_in_thread(_app_write_test, timeout=30),
        "pandas": run_in_thread(_pandas_pipeline_test, timeout=30),
        "packages": run_in_thread(_installed_packages_sample, timeout=50),
        "network": asyncio.create_task(check_network()),
        "llm_keys_models": asyncio.create_task(check_llm_keys_models()),
    }

    if full or RUN_LONGER_CHECKS:
        tasks["duckdb"] = asyncio.create_task(check_duckdb())
        tasks["playwright"] = asyncio.create_task(check_playwright())

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
    failed = [k for k, v in results.items() if v.get("status") != "ok"]
    report["status"] = "warning" if failed else "ok"
    report["summary"]["failed_checks"] = failed
    report["elapsed_seconds"] = (datetime.utcnow() - started).total_seconds()
    return report


# =============================
# Dev Entrypoint
# =============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
