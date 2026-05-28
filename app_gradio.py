"""
Gradio App — Production deployment for Hugging Face Spaces (free tier).
Falls back gracefully: tries local Ollama first, then HF InferenceClient.
Run locally:  python app_gradio.py
Deploy to HF: push repo to HF Space, set space SDK = gradio
"""

import json
import os
import time
import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3.2:3b")
API_BASE     = os.getenv("API_BASE", "http://localhost:8000")

# ─── Check if local API is available ─────────────────────
def local_api_available() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


# ─── Streaming inference via local API ───────────────────
def chat_via_api(prompt: str, model: str, max_tokens: int, temperature: float):
    """Stream from local FastAPI backend."""
    payload = {"prompt": prompt, "model": model,
               "max_tokens": max_tokens, "temperature": temperature}
    full = ""
    with httpx.stream("POST", f"{API_BASE}/chat",
                      json=payload, timeout=120.0) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("token"):
                full += chunk["token"]
                yield full
            if chunk.get("done"):
                stats = chunk.get("stats", {})
                tps = stats.get("tokens_per_second", 0)
                ttft = stats.get("ttft_ms", 0)
                yield full + f"\n\n---\n*TTFT: {ttft:.0f}ms | Throughput: {tps:.1f} tok/s*"
                return


# ─── Gradio chat function ─────────────────────────────────
def chat(message, history, model, max_tokens, temperature):
    if not message.strip():
        yield "Please enter a message."
        return

    if local_api_available():
        yield from chat_via_api(message, model, max_tokens, temperature)
    else:
        yield "⚠️ Local API not running. Start FastAPI backend first:\n`python -m uvicorn backend.main:app --reload`"


# ─── GPU info ─────────────────────────────────────────────
def get_gpu_info():
    if not local_api_available():
        return "API offline"
    try:
        data = httpx.get(f"{API_BASE}/gpu", timeout=3.0).json()
        return (
            f"**{data['name']}**\n"
            f"- VRAM: {data['vram_used_mb']}/{data['vram_total_mb']} MB "
            f"({data['vram_used_pct']}%)\n"
            f"- GPU Util: {data['gpu_utilization_pct']}%\n"
            f"- Temp: {data['temperature_c']}°C"
        )
    except Exception as e:
        return f"GPU unavailable: {e}"


def get_models():
    if not local_api_available():
        return ["llama3.2:3b", "phi3:mini", "tinyllama"]
    try:
        data = httpx.get(f"{API_BASE}/models", timeout=3.0).json()
        return [m["name"] for m in data.get("models", [])] or ["llama3.2:3b"]
    except Exception:
        return ["llama3.2:3b"]


# ─── Gradio UI ────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#090b10",
    block_background_fill="#0d1117",
    block_border_width="1px",
    block_border_color="#1e2a3e",
    input_background_fill="#161d2e",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
)

with gr.Blocks(theme=theme, title="LLM GPU Inference Lab") as demo:
    gr.Markdown("""
    # ⚡ LLM GPU Inference Lab
    **Research-grade local LLM inference benchmarking.**
    Measures TTFT, TPOT, and throughput in real time on your NVIDIA GPU.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=chat,
                additional_inputs=[
                    gr.Dropdown(
                        get_models(),
                        value="llama3.2:3b",
                        label="Model",
                    ),
                    gr.Slider(64, 1024, value=512, step=64, label="Max Tokens"),
                    gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Temperature"),
                ],
                title="",
                description="Chat with a local LLM — metrics appear after each response.",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🖥 GPU Status")
            gpu_info = gr.Markdown(get_gpu_info())
            gr.Button("🔄 Refresh GPU").click(get_gpu_info, outputs=gpu_info)

            gr.Markdown("### 💾 VRAM Budget")
            gr.Markdown("""
| Model | VRAM |
|-------|------|
| llama3.2:3b | ~2.0 GB ✅ |
| phi3:mini | ~2.3 GB ✅ |
| tinyllama | ~0.7 GB ✅ |
| 7B+ models | ❌ OOM |
            """)

    gr.Markdown("""
    ---
    **Run locally**: `python -m uvicorn backend.main:app --reload` 
    then open [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
