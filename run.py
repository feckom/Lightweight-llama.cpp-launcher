#!/usr/bin/env python3
"""
run.py  —  launcher for llama.cpp  (server + CLI mode)

New in this version:
  - Think/reasoning model detection: if the model name or metadata suggests
    a reasoning model (DeepSeek-R1, QwQ, Qwen3-thinking, etc.), the user is
    asked whether to show <think>...</think> blocks.  Default answer is NO
    because raw CoT output is usually noisy and unwanted in production use.
    In server mode this is passed as --reasoning-format none/raw.
    In CLI mode it controls --no-thinking / absence of that flag.
  - Web interface question now defaults to YES: just pressing Enter launches
    the web UI.  The question still waits for a full answer — Enter just
    accepts the default.

Requirements: Python 3.8+, no third-party packages.
"""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS AND GLOBAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR  = Path(__file__).parent.resolve()
MODEL_DIR   = SCRIPT_DIR / "models"
# Executable discovery: try .exe (Windows) then bare name (macOS/Linux).
# This makes the launcher work on all platforms without editing the script.
def _find_exe(stem: str) -> Path:
    """Return the first existing path for stem.exe or stem, else stem.exe."""
    for name in (f"{stem}.exe", stem):
        p = SCRIPT_DIR / name
        if p.exists():
            return p
    return SCRIPT_DIR / f"{stem}.exe"   # fallback (triggers missing-file check)

SERVER_EXE     = _find_exe("llama-server")
CLI_EXE        = _find_exe("llama-cli")
RPC_SERVER_EXE = _find_exe("llama-rpc-server")
LOG_FILE    = SCRIPT_DIR / "run.log"
SERVER_LOG  = SCRIPT_DIR / "server.log"
CONFIG_FILE = SCRIPT_DIR / "config.json"

# Runtime defaults.  All of these can be overridden by config.json.
DEFAULTS = {
    "threads":       6,
    "threads_batch": 12,
    "host":          "127.0.0.1",
    "port":          8080,
    "ready_timeout": 120,   # seconds to wait for /health before warning
    "rpc_port":      50052,  # default port for llama-rpc-server worker
}

# Architecture tags searched in llama-cli --info output and filenames.
# Order matters for substring fallback: more specific tags should come first
# so "mistral" doesn't accidentally match inside a longer model name string.
# qwen35 must come before "qwen" so the more specific tag matches first
ARCH_TAGS = ["gemma", "mistral", "phi", "qwen35", "qwen", "llama", "falcon", "mpt"]

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING DEFAULTS PER ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
#
# These are the LOWEST-priority defaults — they are overridden by:
#   1. Values found in the model's own metadata (llama-cli --info)
#   2. Keys in config.json  (sampling_temp, sampling_top_k, …)
#
# Sources for each architecture:
#   gemma   — Google's Gemma inference guide (temp=1.0, top_k=64)
#   llama   — Meta's recommended settings for Llama 3.x (temp=0.6)
#   mistral — Mistral AI default (temp=0.7, top_k=50)
#   phi     — Microsoft Phi-3/4 docs recommend temp=0 for deterministic output
#   qwen    — Alibaba Qwen2.5 defaults (temp=0.7, top_k=20, top_p=0.8)
#   falcon  — TII defaults
#   mpt     — MosaicML defaults

ARCH_SAMPLING_DEFAULTS = {
    "gemma":    dict(temp=1.0,  top_k=64,  top_p=0.95, repeat_penalty=1.0,  chat_template="gemma"),
    "llama":    dict(temp=0.6,  top_k=40,  top_p=0.9,  repeat_penalty=1.1,  chat_template="chatml"),
    "mistral":  dict(temp=0.7,  top_k=50,  top_p=0.9,  repeat_penalty=1.05, chat_template="mistral"),
    "phi":      dict(temp=0.0,  top_k=50,  top_p=0.95, repeat_penalty=1.0,  chat_template="chatml"),
    # qwen35 = Qwen3 / Qwen3.5 hybrid SSM-attention architecture.
    # Alibaba recommends temp=0.6, top_p=0.95, top_k=20 for thinking mode;
    # for non-thinking use temp=0.7, top_k=20, top_p=0.8.  We use the
    # non-thinking defaults here since thinking is off by default.
    "qwen35":   dict(temp=0.7,  top_k=20,  top_p=0.8,  repeat_penalty=1.0,  chat_template="qwen3"),
    "qwen":     dict(temp=0.7,  top_k=20,  top_p=0.8,  repeat_penalty=1.05, chat_template="chatml"),
    "falcon":   dict(temp=0.8,  top_k=40,  top_p=0.9,  repeat_penalty=1.1,  chat_template="chatml"),
    "mpt":      dict(temp=0.7,  top_k=40,  top_p=0.9,  repeat_penalty=1.1,  chat_template="chatml"),
    "_default": dict(temp=0.7,  top_k=40,  top_p=0.9,  repeat_penalty=1.1,  chat_template="chatml"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# THINK / REASONING MODEL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
#
# "Think" models are reasoning models that emit a hidden chain-of-thought
# inside <think>...</think> XML tags before the final answer.
# Examples: DeepSeek-R1, QwQ-32B, Qwen3 (thinking variants).
#
# In llama.cpp:
#   Server mode:  --reasoning-format none  hides <think> blocks (default here)
#                 --reasoning-format raw   shows raw <think> content
#   CLI mode:     --no-thinking            suppresses <think> output
#                 (omitting the flag)      shows <think> output
#
# We detect think models by matching known substrings in the filename.
# This is intentionally a filename-only heuristic — llama-cli --info does
# not yet expose a stable "is_reasoning_model" metadata key.

_THINK_MODEL_PATTERNS = re.compile(
    # Word boundaries prevent false positives:
    #   \bdeepseek.?r\d\b  — "DeepSeek-R1", "deepseek-r2" (not "nodeepseek")
    #   \bqwq\b             — exact "QwQ" token
    #   \bqwen\d[^/\\]*think — Qwen3/3.5 + "think"/"thinking" anywhere after
    #                          [^/\\]* excludes path separators (safe for rglob)
    #   \br1\b              — standalone "r1" token
    #                          Old: r1(?:[-_.]|$) missed "model-r1.gguf"
    #                          \b correctly handles all separators
    r'\bdeepseek.?r\d\b|\bqwq\b|\bqwen\d[^/\\\\]*think|\br1\b',
    re.IGNORECASE
)

def is_think_model(model_path: Path, model_info: str) -> bool:
    """
    Return True if the model appears to be a reasoning/think model.

    Checks (in order):
      1. Filename regex match against known think-model naming patterns
      2. Presence of 'thinking' or 'reasoning' in the llama-cli --info output
         (future-proofing: llama.cpp may expose this in metadata later)

    Note: even non-think models can emit stray <think> tags if the build
    supports the reasoning-format flag.  We therefore always pass
    --reasoning-format none unless the user explicitly asks to see thinking.
    This function only controls whether we ASK the user about it.
    """
    if _THINK_MODEL_PATTERNS.search(model_path.name):
        return True
    if model_info:
        info_low = model_info.lower()
        if "thinking" in info_low or "reasoning_model" in info_low:
            return True
    return False

def ask_show_thinking(think_detected: bool) -> bool:
    """
    Ask the user whether to show <think> blocks.

    We ALWAYS ask this question when the build supports --reasoning-format,
    but the wording differs:
      - Known think model  → explicit warning that <think> blocks exist
      - Other model        → brief note in case the model surprises the user

    Default is NO (pressing Enter hides them) in both cases.
    Returns True if the user wants to see <think> output, False otherwise.
    """
    print()
    if think_detected:
        print("  This appears to be a reasoning/think model.")
        print("  Think models emit <think>...</think> blocks before their answer.")
    else:
        print("  Some models emit <think>...</think> reasoning blocks.")
    print("  By default these are hidden for a cleaner chat experience.")
    raw = input("  Show <think> reasoning blocks? (y/N, default=N): ").strip().lower()
    return raw == "y"

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING CONFIG: three-layer resolution
# ═══════════════════════════════════════════════════════════════════════════════

def get_sampling_defaults(arch: str, model_info: str, cfg: dict) -> dict:
    """
    Build the final sampling parameter dict by merging three sources.

    Priority (lowest → highest):
      1. ARCH_SAMPLING_DEFAULTS[arch]   — architecture family defaults
      2. parse_sampling_from_info()     — values embedded in model metadata
      3. config.json sampling_* keys   — explicit user override

    This means a Gemma model always starts with temp=1.0 unless its own
    metadata says otherwise, and the user can always pin values in config.json
    without touching the script.
    """
    # Layer 1: architecture defaults
    base = dict(ARCH_SAMPLING_DEFAULTS.get(arch, ARCH_SAMPLING_DEFAULTS["_default"]))

    # Layer 2: model metadata hints (only keys actually found are merged)
    base.update(parse_sampling_from_info(model_info))

    # Layer 3: config.json — accepts both "sampling_temp" and plain "temp"
    for key in ("temp", "top_k", "top_p", "repeat_penalty", "chat_template"):
        for cfg_key in (f"sampling_{key}", key):
            if cfg_key in cfg:
                base[key] = cfg[cfg_key]
                break

    return base

def parse_sampling_from_info(info_text: str) -> dict:
    """
    Extract sampling hints from the raw llama-cli --info text.
    llama.cpp may print lines like:
        general.recommended_temperature = 0.8
        tokenizer.ggml.top_k = 50
    We capture those with named regex patterns and return only found keys,
    so absent keys do not silently overwrite the architecture defaults.
    """
    result = {}
    if not info_text:
        return result

    patterns = {
        "temp":           re.compile(r"(?:recommended_temperature|temperature)\s*=\s*([\d.]+)", re.I),
        "top_k":          re.compile(r"top[_-]k\s*=\s*(\d+)", re.I),
        "top_p":          re.compile(r"top[_-]p\s*=\s*([\d.]+)", re.I),
        "repeat_penalty": re.compile(r"repeat[_-]penalty\s*=\s*([\d.]+)", re.I),
    }

    for key, pattern in patterns.items():
        m = pattern.search(info_text)
        if m:
            try:
                val = m.group(1)
                # Keep integers as int (top_k), floats as float (temp, top_p)
                result[key] = float(val) if "." in val else int(val)
            except ValueError:
                pass

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

_log_lock = threading.Lock()   # prevents interleaved writes from bg threads

def init_logs():
    """
    Truncate (reset) both run.log and server.log at startup.
    Opening with "w" discards any content from previous runs, which prevents
    both files from growing indefinitely across sessions.
    run.log gets a header; server.log starts empty (llama-server writes its own).
    """
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("===== llama.cpp launcher log =====\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    # Truncate server.log — llama-server will write its own startup header
    open(SERVER_LOG, "w").close()

def log(msg: str, also_print: bool = True):
    """
    Write a line to run.log.  Thread-safe via _log_lock.
    also_print=True  → also print to console  (default, for user-visible events)
    also_print=False → log silently           (for debug/internal details)
    """
    line = msg.rstrip()
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    if also_print:
        print(line)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG.JSON
# ═══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    """
    Load optional config.json from the same directory as this script.
    Missing file is silently ignored — all keys fall back to DEFAULTS.

    Supported keys:
        threads, threads_batch, host, port, ready_timeout
        ctx, ngl, batch, ubatch           (parameter overrides)
        sampling_temp, sampling_top_k,
        sampling_top_p, sampling_repeat_penalty,
        sampling_chat_template            (sampling overrides)

    Example config.json:
        {
            "threads": 8,
            "port": 8081,
            "ctx": 8192,
            "sampling_temp": 0.5,
            "sampling_chat_template": "llama3"
        }
    """
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                overrides = json.load(f)
            cfg.update(overrides)
            print(f"  [config] Loaded {CONFIG_FILE.name}: {overrides}")
        except Exception as e:
            print(f"  [config] Could not read {CONFIG_FILE.name}: {e}")
    return cfg

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL METADATA  (single llama-cli --info call)
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_info(model_path: Path) -> str:
    """
    Run  llama-cli --model <path> --info  exactly once and return the combined
    stdout+stderr as a raw string.

    This output is then consumed by:
      - parse_architecture()        → architecture family (gemma, llama, …)
      - parse_ctx_hint()            → model's trained context length
      - parse_sampling_from_info()  → recommended sampling parameters
      - is_think_model()            → reasoning model detection

    Running --info only once avoids spawning llama-cli 4 separate times for
    the same model file, which would add several seconds of startup overhead.

    Returns "" if llama-cli is absent or the call fails for any reason.
    """
    if not CLI_EXE.exists():
        return ""
    try:
        result = subprocess.run(
            [str(CLI_EXE), "--model", str(model_path), "--info"],
            capture_output=True, text=True, timeout=20,
            cwd=str(SCRIPT_DIR)
        )
        return result.stdout + result.stderr
    except Exception as e:
        log(f"  [WARN] llama-cli --info failed: {e}", also_print=False)
        return ""

def parse_architecture(info_text: str, model_path: Path) -> str:
    """
    Determine model architecture in four passes (most → least reliable):

    Pass 1 — explicit key=value line in --info output
        Looks for  general.architecture = gemma  or  arch: gemma.
        Most reliable: this is the value llama.cpp itself parsed from GGUF.
        Regex prevents false positives from tool/build metadata that might
        contain another architecture name as a substring.

    Pass 2 — full-text substring scan of --info output
        Catches architectures mentioned anywhere in the output when not on
        a dedicated key line.

    Pass 3 — binary GGUF header (first 8 KB)
        Reads raw bytes; architecture string appears in metadata section.
        Works even when llama-cli is absent.

    Pass 4 — filename substring
        Last resort; works for well-named community model files.
    """
    if info_text:
        # Pass 1: structured key=value — two styles from llama-cli --info:
        #
        # Style A — kv metadata dump (most reliable, appears first):
        #   "llama_model_loader: - kv 0: general.architecture str = qwen35"
        #   The GGUF type annotation ("str", "u32" …) sits between the key
        #   and "=". [^=:\n]* skips it. Without this fix the old regex
        #   captured "str" instead of "qwen35" — root cause of "arch: llama".
        #
        # Style B — print_info summary:
        #   "print_info: arch                  = qwen35"  or  "arch = qwen35"
        #
        # group(1) = Style A value, group(2) = Style B value
        kv = re.compile(
            r"general\.architecture[^=:\n]*[=:]\s*(\w+)"
            r"|(?:print_info:\s*)?\barch\s*[=:]\s*(\w+)",
            re.I
        )
        for line in info_text.splitlines():
            m = kv.search(line)
            if m:
                val = (m.group(1) or m.group(2) or "").lower()
                for arch in ARCH_TAGS:
                    if arch in val:
                        return arch

        # Pass 2: full-text scan
        low = info_text.lower()
        for arch in ARCH_TAGS:
            if arch in low:
                return arch

    # Pass 3: GGUF binary header
    try:
        with open(model_path, "rb") as f:
            data = f.read(8192)
        text = data.decode("utf-8", errors="ignore").lower()
        for arch in ARCH_TAGS:
            if arch in text:
                return arch
    except Exception as e:
        log(f"  [WARN] Header read failed: {e}", also_print=False)

    # Pass 4: filename
    name = model_path.name.lower()
    for arch in ARCH_TAGS:
        if arch in name:
            return arch

    return "unknown"

def parse_ctx_hint(info_text: str) -> int:
    """
    Extract the model's trained context length from --info output.
    Matches  context_length = 8192  or  n_ctx_train = 8192  style lines.
    Returns 0 if not found (caller treats 0 as "no cap available").
    """
    if not info_text:
        return 0
    m = re.search(
        # Match both:
        #   print_info: n_ctx_train           = 262144
        #   kv dump:    qwen35.context_length u32              = 262144
        # The optional  (?:\w+\s+)?  handles the GGUF type annotation (u32, i32…)
        r"(?:context[_-]length|n_ctx_train)\s+(?:\w+\s+)?[=:]\s*(\d+)",
        info_text, re.I
    )
    if m:
        val = int(m.group(1))
        if 512 <= val <= 131072:
            return val
    return 0

# ═══════════════════════════════════════════════════════════════════════════════
# VRAM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_vram_via_nvidia_smi() -> tuple:
    """
    Query nvidia-smi for per-GPU VRAM totals and free memory.

    Returns (max_total_mb, list_of_gpu_dicts).
    GPU indices may be non-sequential (e.g. [0, 2] when a GPU is hidden
    via CUDA_VISIBLE_DEVICES in the parent environment) — we preserve the
    exact index values reported by nvidia-smi rather than re-numbering them.

    Returns (0, []) when nvidia-smi is absent or fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8
        )
        if result.returncode != 0:
            return 0, []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index":    int(parts[0]),
                    "name":     parts[1],
                    "total_mb": int(parts[2]),
                    "free_mb":  int(parts[3]),
                })
        if gpus:
            return max(g["total_mb"] for g in gpus), gpus
    except FileNotFoundError:
        pass   # nvidia-smi not installed — fall through to WMI
    except Exception as e:
        log(f"  [WARN] nvidia-smi failed: {e}", also_print=False)
    return 0, []

def get_vram_via_wmi() -> int:
    """
    Fallback VRAM query via Windows WMI (Win32_VideoController.AdapterRAM).
    This value is often capped at 4 GB by drivers and may include shared
    system memory — treat it as a rough lower-bound estimate only.
    Returns 0 on failure.
    """
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController | "
             "Measure-Object -Property AdapterRAM -Maximum).Maximum / 1MB"],
            capture_output=True, text=True, timeout=10
        )
        val = result.stdout.strip()
        if val:
            return int(float(val))
    except Exception:
        pass
    return 0

def get_vram_info() -> tuple:
    """
    Returns (max_vram_mb, gpu_list).
    Tries nvidia-smi first; falls back to WMI; returns (0, []) on CPU-only.
    """
    vram_mb, gpus = get_vram_via_nvidia_smi()
    if vram_mb > 0:
        return vram_mb, gpus
    vram_mb = get_vram_via_wmi()
    if vram_mb > 0:
        return vram_mb, [{"index": 0, "name": "GPU (WMI)",
                          "total_mb": vram_mb, "free_mb": vram_mb}]
    return 0, []

def select_gpu(gpus: list) -> int:
    """
    When only one GPU is present, return its index immediately without prompting.

    For multiple GPUs:
      - Display each GPU with its ACTUAL index from nvidia-smi (may be non-sequential)
      - Validate the user's input against the set of known indices
      - Fall back to the first GPU in the list on invalid input instead of crashing

    The index is later used both for --device <value> and CUDA_VISIBLE_DEVICES.
    """
    if len(gpus) <= 1:
        return gpus[0]["index"] if gpus else 0

    valid_indices = {g["index"] for g in gpus}
    print()
    print("===== Multiple GPUs detected =====")
    for g in gpus:
        print(f"  {g['index']}) {g['name']}  —  "
              f"{g['total_mb']} MB total / {g['free_mb']} MB free")
    print(f"  Valid indices: {sorted(valid_indices)}")
    print()

    raw = input("Select GPU index: ").strip()
    try:
        choice = int(raw)
        if choice in valid_indices:
            return choice
        print(f"  [WARN] Index {choice} not in list — using GPU {gpus[0]['index']}.")
    except ValueError:
        print(f"  [WARN] Invalid input — using GPU {gpus[0]['index']}.")

    return gpus[0]["index"]   # safe fallback

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTISATION DETECTION AND VRAM ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# We estimate how much VRAM the model needs for full GPU offload by multiplying
# the file size by a quantisation-specific ratio.
#
# The regex requires the quant tag to be preceded and followed by a separator
# character (underscore, hyphen, dot) or end-of-string.  This prevents
# partial matches like "q4" inside "q4something_weird".
#
# Ratios (bytes per parameter, approximate):
#   IQ2/Q2 ≈ 0.35    Q3/IQ3 ≈ 0.44    Q4/IQ4 ≈ 0.55
#   Q5     ≈ 0.67    Q6     ≈ 0.75    Q8     ≈ 1.00

_QUANT_RE = re.compile(
    r'[_\-.](?P<q>iq[2-4]|q[2-8])(?:_k(?:_[smlx]+)?|_[0-9]+)?(?:[_.\-]|$)',
    re.IGNORECASE
)

QUANT_VRAM_RATIO = {
    "iq2": 0.35, "iq3": 0.44, "iq4": 0.55,
    "q2":  0.35, "q3":  0.44, "q4":  0.55,
    "q5":  0.67, "q6":  0.75, "q8":  1.00,
}

def estimate_vram_needed_mb(model_path: Path) -> int:
    """
    Estimate VRAM required for full GPU offload of this model file.

    Formula:  file_size_MB  ×  quant_ratio  ×  1.15

    The 1.15× overhead factor accounts for:
      - Fixed CUDA library allocations (~100–300 MB)
      - Activation buffers during forward pass
      - KV cache at the default batch size

    Note: KV cache grows linearly with ctx size.  The 1.15× factor assumes
    a moderate ctx.  For very large ctx (>16K) users should reduce ngl manually
    if the server OOMs.  Accurate KV cache math would require parsing n_layers,
    n_heads, and head_dim from GGUF metadata — out of scope for this launcher.
    """
    file_mb = model_path.stat().st_size / (1024 * 1024)
    m = _QUANT_RE.search(model_path.name)
    if m:
        key = m.group("q").lower()
        # IQ-quants share ratios with their Q counterparts
        ratio_key = key if key in QUANT_VRAM_RATIO else key.replace("iq", "q")
        ratio = QUANT_VRAM_RATIO.get(ratio_key, 1.05)
    else:
        ratio = 1.05   # unknown quant → assume near-F16

    return int(file_mb * ratio * 1.15)

# llama.cpp reserves this much free VRAM as a safety margin internally.
# We subtract it from available free VRAM so our fit estimate matches
# what llama.cpp will actually accept without an OOM warning.
_LLAMA_VRAM_SAFETY_MB = 1024

def auto_params(vram_free_mb: int, has_gpu: bool, model_path: Path) -> dict:
    """
    Choose ctx, ngl, batch, ubatch automatically.

    GPU layer count (ngl) is decided by comparing estimated model VRAM need
    against available free VRAM — more accurate than raw VRAM tiers alone:
      ≥100% fits  →  ngl=99  (full offload)
      ≥ 50% fits  →  ngl=50  (partial offload, mostly weight tensors)
      ≥ 25% fits  →  ngl=20  (minimal offload, embedding + first layers only)
      < 25% fits  →  ngl=0   (CPU-only despite GPU present)

    CPU-only mode:  ngl is always 0 (not 20) to avoid confusion — passing
    ngl=20 to a CPU build triggers a warning inside llama.cpp and is misleading.

    We subtract _LLAMA_VRAM_SAFETY_MB (1024 MB) from free VRAM before the fit
    calculation.  llama.cpp tries to keep this much memory free at all times
    and aborts fitting if it cannot.  Without this correction our fit% is
    optimistic and can suggest ngl=99 for models that will trigger the
    "cannot meet free memory target" warning at startup.

    ctx tiers are still VRAM-based because KV cache scales with ctx, not model size.
    """
    if not has_gpu:
        return dict(ctx=2048, ngl=0, batch=512, ubatch=512)

    # Effective free VRAM after llama.cpp safety reserve
    effective_free_mb = max(0, vram_free_mb - _LLAMA_VRAM_SAFETY_MB)
    needed_mb         = estimate_vram_needed_mb(model_path)
    ratio             = effective_free_mb / needed_mb if needed_mb > 0 else 0

    if   ratio >= 1.0:  ngl = 99
    elif ratio >= 0.5:  ngl = 50
    elif ratio >= 0.25: ngl = 20
    else:               ngl = 0

    # ctx tiers use effective_free_mb (with safety margin) for consistency
    if   effective_free_mb >= 12000: ctx, batch, ubatch = 16384, 2048, 2048
    elif effective_free_mb >= 8000:  ctx, batch, ubatch = 8192,  1024, 1024
    elif effective_free_mb >= 4000:  ctx, batch, ubatch = 4096,  512,  512
    else:                            ctx, batch, ubatch = 2048,  512,  512

    return dict(ctx=ctx, ngl=ngl, batch=batch, ubatch=ubatch)

# ═══════════════════════════════════════════════════════════════════════════════
# SERVER CAPABILITY PROBES  (flash-attn + --device style)
# ═══════════════════════════════════════════════════════════════════════════════

def probe_server_help(server_exe: Path) -> str:
    """
    Run  llama-server --help  exactly once and return the combined
    stdout+stderr.  All capability probes (flash-attn style, --device
    style, --reasoning-format support, --no-thinking for CLI) reuse
    this single cached string — no repeated subprocess spawning.

    Previously the script called --help 3–4 times in sequence:
      detect_flash_attn_style()  → --help
      detect_device_style()      → --help
      reasoning-format check     → --help
      --no-thinking check (CLI)  → --help
    This consolidates all of them into one call.
    """
    try:
        result = subprocess.run(
            [str(server_exe), "--help"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout + result.stderr
    except Exception:
        return ""


def detect_flash_attn_style(server_exe: Path, help_text: str = "") -> object:
    """
    Probe which flash-attention flag style this llama-server build uses.

    llama.cpp has had two incompatible flag styles across versions:
      Old builds:  --flash-attn            (boolean switch, no value)
      New builds:  --flash-attn [on|off|auto]  (requires an explicit value)

    Passing the wrong style causes an immediate "expected value for argument"
    crash on startup.  We parse the --help output to detect which style to use
    before ever launching the server.

    help_text: pass the cached --help output from probe_server_help() to avoid
    spawning an extra subprocess.  If empty, falls back to running --help itself.

    Returns:
        None                  — flash-attn not compiled into this build
        "--flash-attn"        — old boolean style
        "--flash-attn auto"   — new value style (we default to "auto")
    """
    if not help_text:
        try:
            result = subprocess.run(
                [str(server_exe), "--help"],
                capture_output=True, text=True, timeout=10
            )
            help_text = result.stdout + result.stderr
        except Exception:
            return None

    if "--flash-attn" not in help_text and "-fa" not in help_text:
        return None   # feature not compiled in

    for line in help_text.splitlines():
        if "--flash-attn" in line or "-fa," in line:
            if "on|off|auto" in line or "[on" in line:
                return "--flash-attn auto"   # new style
            else:
                return "--flash-attn"         # old style

    # Found in --help but style line not matched — default to new style
    # (safer: an unnecessary value is ignored; a missing required value crashes)
    return "--flash-attn auto"

# --device flag syntax varies by backend and build:
#   numeric  →  --device 0          (older CUDA builds)
#   name     →  --device CUDA0      (current llama.cpp CUDA builds)
#   colon    →  --device cuda:0     (some CUDA and Vulkan builds)
#   Vulkan   →  --device Vulkan0    (Vulkan backend by name)
# We detect the style from --help and return a format string.

_DEVICE_PATTERNS = [
    (re.compile(r'--device\s+cuda:\d',    re.I), "cuda:{idx}"),
    (re.compile(r'--device\s+vulkan:\d',  re.I), "Vulkan:{idx}"),
    (re.compile(r'--device\s+CUDA\d'          ), "CUDA{idx}"),
    (re.compile(r'--device\s+Vulkan\d'        ), "Vulkan{idx}"),
    (re.compile(r'--device\s+<int>',      re.I), "{idx}"),
    (re.compile(r'--device',              re.I), "CUDA{idx}"),   # generic fallback
]

def detect_device_style(server_exe: Path, help_text: str = ""):
    """
    Returns a format string like "CUDA{idx}" / "cuda:{idx}" / "{idx}",
    or None if this build does not support --device at all.
    Caller uses  fmt.format(idx=gpu_index)  to build the final flag value.

    help_text: pass the cached --help output from probe_server_help() to avoid
    spawning an extra subprocess.  If empty, runs --help itself.
    """
    if not help_text:
        try:
            result = subprocess.run(
                [str(server_exe), "--help"],
                capture_output=True, text=True, timeout=10
            )
            help_text = result.stdout + result.stderr
        except Exception:
            return None
    if "--device" not in help_text.lower():
        return None
    for pattern, fmt in _DEVICE_PATTERNS:
        if pattern.search(help_text):
            return fmt
    return None

def build_device_args(device_fmt, gpu_index: int) -> list:
    """
    Build the --device argument list (possibly empty).

    For numeric style with index 0 we skip the flag entirely because 0 is
    already the default — passing it explicitly adds noise to the command line.
    For all other styles / indices we always pass the flag so the correct
    GPU is selected even when multiple are present.
    """
    if device_fmt is None:
        return []
    if device_fmt == "{idx}" and gpu_index == 0:
        return []
    return ["--device", device_fmt.format(idx=gpu_index)]

# ═══════════════════════════════════════════════════════════════════════════════
# PORT CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def port_in_use(host: str, port: int) -> bool:
    """
    Return True if something is already listening on host:port.
    A connect attempt is faster and more reliable than parsing netstat output.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT STREAMING
# ═══════════════════════════════════════════════════════════════════════════════

def stream_output(stream, log_path: Path, stop_event: threading.Event):
    """
    Read lines from a subprocess stdout/stderr stream, print each line to the
    console in real time, and append it to the server log file.

    Runs on a dedicated daemon thread so the main thread remains free to poll
    /health.  stop_event is checked at the start of each iteration so the
    thread exits cleanly when the server stops or Ctrl+C is pressed.

    server.log was already truncated by init_logs() so we open in append mode
    here.  Using a with-block ensures the file handle is closed on exception.
    """
    try:
        with open(log_path, "a", encoding="utf-8", errors="replace") as lf:
            for line in stream:
                if stop_event.is_set():
                    break
                text = line.rstrip()
                print(text)
                lf.write(text + "\n")
                lf.flush()
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def wait_for_health(host: str, port: int, timeout: int,
                    proc: subprocess.Popen,
                    stop_event: threading.Event) -> bool:
    """
    Poll  GET /health  once per second until HTTP 200 is received or timeout.

    On each iteration (before the HTTP call):
      - Check stop_event: if set, raise KeyboardInterrupt to propagate Ctrl+C
      - Check proc.poll(): if the server process has already exited, abort
        immediately with a clear error message pointing to server.log.
        This converts a 120-second timeout into a ~1-second failure, which
        makes crash diagnosis much faster.

    Returns True on success (server ready), False on timeout or early crash.
    Raises KeyboardInterrupt when stop_event is set by the signal handler.
    """
    url = f"http://{host}:{port}/health"
    for i in range(1, timeout + 1):

        if stop_event.is_set():
            raise KeyboardInterrupt

        # Early crash detection — check BEFORE making the HTTP request
        if proc.poll() is not None:
            print()
            log(f"[ERROR] Server exited early (code {proc.returncode}).")
            log(f"        Check {SERVER_LOG.name} for details.")
            return False

        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    print()
                    log(f"[OK] Server ready after {i} second(s).")
                    return True
        except Exception:
            pass   # connection refused, timeout, etc. — keep polling

        pct = int(i * 100 / timeout)
        bar = ("=" * int(i * 30 / timeout)).ljust(30)
        print(f"  Loading [{bar}] {pct}%   ", end="\r", flush=True)
        time.sleep(1)

    print()
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# SERVER MODE
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# RPC WORKER MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_rpc_worker(cfg: dict, gpu_index: int):
    """
    Launch this machine as an RPC worker node using llama-rpc-server.

    In RPC (router) mode, llama.cpp splits model layers across multiple
    machines over the network.  The main server runs on one machine and
    offloads GPU layers to one or more RPC worker nodes.

    Worker node role:
      - Does NOT load a model file
      - Exposes local GPU memory as a remote backend
      - The main server connects via --rpc host:port and assigns layers to it

    Typical setup:
      Machine B (this, worker):   python run.py  → choose "worker"
      Machine A (main server):    python run.py  → choose "router", enter B's IP

    The worker listens on all interfaces (0.0.0.0) so the main server can
    reach it.  Change rpc_host to a specific interface if needed for security.
    """
    if not RPC_SERVER_EXE.exists():
        log(f"[ERROR] {RPC_SERVER_EXE.name} not found. "
            "Download it from the llama.cpp release page.")
        return

    rpc_port = cfg.get("rpc_port", 50052)
    rpc_host = cfg.get("rpc_host", "0.0.0.0")   # listen on all interfaces

    cmd = [
        str(RPC_SERVER_EXE),
        "--host", rpc_host,
        "--port", str(rpc_port),
    ]
    # Pass GPU device if not the default GPU 0
    if gpu_index != 0:
        cmd.extend(["--device", str(gpu_index)])

    log(f"RPC worker command: {' '.join(cmd)}", also_print=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    print()
    print("-" * 70)
    print(f"  RPC worker listening on  {rpc_host}:{rpc_port}")
    print(f"  Other machines connect with:  --rpc <YOUR_IP>:{rpc_port}")
    print("  Ctrl+C to stop")
    print("-" * 70)
    print()

    stop_event = threading.Event()
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, cwd=str(SCRIPT_DIR), env=env,
        )
        log(f"RPC worker PID: {proc.pid}", also_print=False)

        reader = threading.Thread(
            target=stream_output,
            args=(proc.stdout, SERVER_LOG, stop_event),
            daemon=True
        )
        reader.start()
        proc.wait()
        stop_event.set()
        reader.join(timeout=5)
        log(f"RPC worker exited with code {proc.returncode}.")

    except KeyboardInterrupt:
        print()
        log("RPC worker interrupted — stopping...")
        stop_event.set()
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log("RPC worker stopped.")

def run_server(model: Path, params: dict, sampling: dict,
               fa_flag, device_fmt, show_thinking: bool,
               reasoning_fmt_supported: bool,
               cfg: dict, gpu_index: int,
               rpc_servers: str = ""):
    """
    Launch llama-server, stream its output to console + server.log,
    poll /health, and open the browser when the server is ready.

    think model handling:
      --reasoning-format none  →  strip <think> blocks from API responses
      --reasoning-format raw   →  pass <think> blocks through unchanged
    The flag is only appended if the build supports it (detected via --help).
    """
    host = cfg["host"]
    port = cfg["port"]

    if port_in_use(host, port):
        log(f"[ERROR] Port {port} already in use. "
            f"Stop the other process or set 'port' in config.json.")
        return

    cmd = [
        str(SERVER_EXE),
        "--model",          str(model),
        "--ctx-size",       str(params["ctx"]),
        "--n-gpu-layers",   str(params["ngl"]),
        "--batch-size",     str(params["batch"]),
        "--ubatch-size",    str(params["ubatch"]),
        "--threads",        str(cfg["threads"]),
        "--threads-batch",  str(cfg["threads_batch"]),
        "--temp",           str(sampling["temp"]),
        "--top-k",          str(int(sampling["top_k"])),
        "--top-p",          str(sampling["top_p"]),
        "--repeat-penalty", str(sampling["repeat_penalty"]),
        "--host",           host,
        "--port",           str(port),
        "--chat-template",  sampling["chat_template"],
    ]

    # --device: use build-appropriate syntax, skip if not supported
    cmd.extend(build_device_args(device_fmt, gpu_index))

    # --rpc: connect to remote RPC worker nodes for layer offloading.
    # rpc_servers is a comma-separated list of host:port addresses.
    # Example:  "192.168.1.10:50052,192.168.1.11:50052"
    # Workers must be running llama-rpc-server before this server starts.
    if rpc_servers:
        cmd.extend(["--rpc", rpc_servers])
        log(f"RPC backends: {rpc_servers}", also_print=False)

    # --flash-attn: append in the correct style for this build
    if fa_flag:
        cmd.extend(fa_flag.split())

    # --reasoning-format: supported status passed in from main (already probed).
    # ALWAYS add this flag when supported — not only for think models.
    # Default is "none" which hides <think> tags for ALL models, preventing
    # stray reasoning blocks from appearing in the web UI unexpectedly.
    # "raw" passes them through when the user explicitly requested it.
    if reasoning_fmt_supported:
        rf = "raw" if show_thinking else "none"
        cmd.extend(["--reasoning-format", rf])
        log(f"reasoning-format: {rf}", also_print=False)

    log(f"Command: {' '.join(cmd)}", also_print=False)

    # CUDA_VISIBLE_DEVICES as secondary hint for older builds without --device
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    print()
    print("-" * 70)
    print(f"  llama-server output  (log: {SERVER_LOG.name})")
    print("  Ctrl+C to stop")
    print("-" * 70)
    print()

    stop_event = threading.Event()
    proc       = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → single stream
            text=True, encoding="utf-8", errors="replace",
            bufsize=1,                  # line-buffered for real-time output
            cwd=str(SCRIPT_DIR), env=env,
        )
        log(f"PID: {proc.pid}", also_print=False)

        # Background thread: reads server output → console + server.log
        reader = threading.Thread(
            target=stream_output,
            args=(proc.stdout, SERVER_LOG, stop_event),
            daemon=True
        )
        reader.start()

        # Poll /health — server output continues appearing during this wait
        try:
            ready = wait_for_health(host, port, cfg["ready_timeout"],
                                    proc, stop_event)
        except KeyboardInterrupt:
            raise   # re-raised to the outer except block below

        if not ready:
            if proc.poll() is None:
                # Server is still running but not responding — timeout case
                print()
                log("[WARNING] Server not responding within timeout.")
                print(f"  Check {SERVER_LOG.name} for error details.")
                if input("Open browser anyway? (y/n): ").strip().lower() != "y":
                    log("Launch cancelled.")
                    stop_event.set()
                    proc.terminate()
                    return
            else:
                # Process already exited — error reported inside wait_for_health
                return
        else:
            time.sleep(0.5)   # brief pause to let the server stabilise

        url = f"http://{host}:{port}"
        log(f"Opening browser: {url}")
        os.startfile(url)

        print()
        print(f"  Server : {url}")
        print(f"  Logs   : {LOG_FILE.name}  |  {SERVER_LOG.name}")
        print("  Ctrl+C to stop.")

        # Show a note about <think> visibility ONLY for confirmed think/reasoning
        # models (DeepSeek-R1, QwQ, Qwen3-thinking …).  For regular models the
        # note is irrelevant and just adds noise to the output.
        #
        # Background: --reasoning-format none was passed (confirmed by
        # "thinking = 0" in server.log).  The built-in llama.cpp web UI in
        # build b8369 renders the raw token stream and shows <think> blocks
        # despite the server-side setting.  This is a known UI limitation.
        # Fix: update llama.cpp (>= ~b4100) or use Open WebUI.
        if think_detected and show_thinking is False and reasoning_fmt_supported:
            print()
            print("  NOTE: <think> blocks are suppressed server-side")
            print("        (--reasoning-format none).  If they still appear")
            print("        in the web UI, update llama.cpp or use Open WebUI:")
            print("        https://github.com/open-webui/open-webui")
        print()

        proc.wait()            # block here until server exits normally
        stop_event.set()
        reader.join(timeout=5)
        log(f"Server exited with code {proc.returncode}.")

    except KeyboardInterrupt:
        print()
        log("Interrupted — stopping server...")
        stop_event.set()
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()   # force-kill if terminate() didn't work in time
        log("Server stopped.")

# ═══════════════════════════════════════════════════════════════════════════════
# CLI MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_cli(model: Path, params: dict, sampling: dict,
            fa_flag, device_fmt, show_thinking: bool,
            reasoning_fmt_supported: bool,
            cfg: dict, gpu_index: int,
            cli_help_text: str = ""):
    """
    Launch llama-cli in interactive mode directly in this terminal window.
    stdin/stdout/stderr are inherited (not redirected) so the user types
    directly into the model — no streaming wrapper needed.

    think model handling:
      --no-thinking  →  suppress <think> blocks in terminal output
      (no flag)      →  show <think> blocks
    """
    cmd = [
        str(CLI_EXE),
        "--model",          str(model),
        "--ctx-size",       str(params["ctx"]),
        "--n-gpu-layers",   str(params["ngl"]),
        "--batch-size",     str(params["batch"]),
        "--ubatch-size",    str(params["ubatch"]),
        "--threads",        str(cfg["threads"]),
        "--threads-batch",  str(cfg["threads_batch"]),
        "--temp",           str(sampling["temp"]),
        "--top-k",          str(int(sampling["top_k"])),
        "--top-p",          str(sampling["top_p"]),
        "--repeat-penalty", str(sampling["repeat_penalty"]),
        "--chat-template",  sampling["chat_template"],
    ]

    cmd.extend(build_device_args(device_fmt, gpu_index))

    if fa_flag:
        cmd.extend(fa_flag.split())

    # Probe llama-cli --help for optional flags.
    # We probe CLI separately from the server because the two binaries may
    # differ in supported flags across builds and platforms.
    # cli_help_text is passed in from main() to avoid a duplicate subprocess.
    _cli_help = cli_help_text or ""
    if not _cli_help:
        try:
            r = subprocess.run(
                [str(CLI_EXE), "--help"],
                capture_output=True, text=True, timeout=10
            )
            _cli_help = r.stdout + r.stderr
        except Exception:
            pass

    # --interactive: present in older llama-cli builds, removed in newer ones
    # where interactive mode is the default behaviour when no prompt is given.
    # Passing it to a build that dropped it causes immediate exit code 1.
    # Only add it if --help lists it explicitly.
    if "--interactive" in _cli_help:
        cmd.append("--interactive")

    # --no-thinking: suppress <think> blocks in terminal output.
    if not show_thinking and "--no-thinking" in _cli_help:
        cmd.append("--no-thinking")

    log(f"Command: {' '.join(cmd)}", also_print=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    print()
    print("-" * 70)
    print("  CLI chat  |  Ctrl+C to exit")
    print("-" * 70)
    print()

    try:
        # subprocess.run (not Popen) so stdin is fully inherited — interactive
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env)
        log(f"CLI exited with code {result.returncode}.")
        if result.returncode != 0:
            print(f"\n[ERROR] llama-cli exited with code {result.returncode}.")
    except KeyboardInterrupt:
        log("CLI interrupted.")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Init ──────────────────────────────────────────────────────────────────
    init_logs()       # truncate both log files before anything else
    cfg = load_config()

    # ── Validate executables and model directory ───────────────────────────────
    if not SERVER_EXE.exists():
        sys.exit(f"[ERROR] {SERVER_EXE.name} not found in {SCRIPT_DIR}")
    if not MODEL_DIR.exists():
        sys.exit(f"[ERROR] models/ directory not found in {SCRIPT_DIR}")
    if not CLI_EXE.exists():
        print(f"[WARNING] {CLI_EXE.name} not found — "
              "CLI mode and architecture detection unavailable.")
        log(f"WARNING: {CLI_EXE.name} missing")

    # ── Discover models ───────────────────────────────────────────────────────
    # rglob instead of glob — finds models in subdirectories too.
    # Covers LM Studio layout (models/publisher/repo/model.gguf),
    # Ollama-style dirs, and any custom folder nesting.
    models = sorted(MODEL_DIR.rglob("*.gguf"))
    if not models:
        sys.exit(f"[ERROR] No .gguf files in {MODEL_DIR}")

    print()
    print("===== Available GGUF models =====")
    print()
    for i, m in enumerate(models, 1):
        size_gb = m.stat().st_size / 1_073_741_824
        print(f"  {i}) {m.name}  (~{size_gb:.1f} GB)")
    print()

    while True:
        try:
            choice = int(input("Select model number: "))
            if 1 <= choice <= len(models):
                break
            print(f"  Enter a number between 1 and {len(models)}.")
        except ValueError:
            print("  Enter a number.")

    model   = models[choice - 1]
    size_gb = model.stat().st_size / 1_073_741_824
    print()
    print(f"Selected  : {model.name}  (~{size_gb:.1f} GB)")
    log(f"Selected: {model}  (~{size_gb:.1f} GB)")

    # ── Single --info call; all metadata parsed from this one output ──────────
    print()
    print("Reading model metadata (llama-cli --info)...")
    model_info = get_model_info(model)

    arch    = parse_architecture(model_info, model)
    ctx_max = parse_ctx_hint(model_info)

    print(f"  Architecture : {arch}")
    print(f"  Max ctx hint : {ctx_max if ctx_max else 'not found'}")
    log(f"Architecture: {arch}")
    if ctx_max:
        log(f"Model max ctx: {ctx_max}")

    # ── Think model detection ─────────────────────────────────────────────────
    # Detect early so we can show the flag in the parameter summary.
    # The actual question is asked AFTER the capability probe below, because
    # we only ask when the build actually supports --reasoning-format.
    think_detected = is_think_model(model, model_info)
    show_thinking  = False   # safe default — may be overwritten after probe
    if think_detected:
        log(f"Think/reasoning model detected: {model.name}")

    # ── VRAM ──────────────────────────────────────────────────────────────────
    print()
    print("Querying GPU VRAM...")
    vram_mb, gpus = get_vram_info()
    has_gpu = vram_mb > 0

    if gpus:
        for g in gpus:
            print(f"  GPU {g['index']}: {g['name']}  "
                  f"{g['total_mb']} MB total / {g['free_mb']} MB free")
    else:
        print("  No GPU detected — CPU-only mode (ngl=0).")
    log(f"VRAM: {vram_mb} MB  GPUs: {len(gpus)}")

    gpu_index = select_gpu(gpus) if gpus else 0
    vram_free = vram_mb
    if len(gpus) > 1:
        log(f"GPU selected: {gpu_index}")
        sel = next((g for g in gpus if g["index"] == gpu_index), None)
        if sel:
            vram_free = sel["free_mb"]

    # ── Parameters ────────────────────────────────────────────────────────────
    params    = auto_params(vram_free, has_gpu, model)
    needed_mb = estimate_vram_needed_mb(model)
    # Use the same effective free VRAM as auto_params (minus llama.cpp safety reserve)
    # so the displayed fit% matches the actual ngl decision
    effective_free = max(0, vram_free - _LLAMA_VRAM_SAFETY_MB)
    fit_pct = int(effective_free / needed_mb * 100) if needed_mb and has_gpu else 0

    # Apply any config.json overrides for raw parameters
    for key in ("ctx", "ngl", "batch", "ubatch"):
        if key in cfg:
            params[key] = cfg[key]

    # Cap ctx to model's trained maximum if metadata provided it
    if ctx_max > 0 and params["ctx"] > ctx_max:
        print(f"  Capping ctx {params['ctx']} → {ctx_max} (model maximum)")
        log(f"ctx capped to {ctx_max}")
        params["ctx"] = ctx_max

    # ── Sampling ──────────────────────────────────────────────────────────────
    sampling = get_sampling_defaults(arch, model_info, cfg)

    # ── Display and confirm ───────────────────────────────────────────────────
    print()
    print("===== Parameters =====")
    print(f"  ctx           : {params['ctx']} tokens")
    print(f"  gpu layers    : {params['ngl']}"
          + (f"  (~{fit_pct}% of model fits in free VRAM)" if has_gpu else " (CPU-only)"))
    print(f"  batch         : {params['batch']}")
    print(f"  ubatch        : {params['ubatch']}")
    print(f"  threads       : {cfg['threads']}")
    print(f"  threads-batch : {cfg['threads_batch']}")
    print()
    print("===== Sampling =====")
    print(f"  template      : {sampling['chat_template']}")
    print(f"  temp          : {sampling['temp']}"
          + ("  (deterministic/greedy)" if sampling["temp"] == 0.0 else ""))
    print(f"  top_k         : {sampling['top_k']}")
    print(f"  top_p         : {sampling['top_p']}")
    print(f"  repeat_penalty: {sampling['repeat_penalty']}")
    if think_detected:
        print(f"  show thinking : {'yes' if show_thinking else 'no (hidden)'}")
    print()
    log(f"ctx={params['ctx']} ngl={params['ngl']} fit={fit_pct}% "
        f"temp={sampling['temp']} template={sampling['chat_template']}")

    confirm = input("Use these parameters? (Enter=yes / n=manual): ").strip().lower()
    if confirm == "n":
        try:
            params["ctx"]    = int(input("  ctx tokens          : "))
            params["ngl"]    = int(input("  GPU layers (0=CPU)  : "))
            params["batch"]  = int(input("  batch size          : "))
            params["ubatch"] = int(input("  ubatch size         : "))
            t = input(f"  temp  [{sampling['temp']}]  : ").strip()
            if t: sampling["temp"] = float(t)
            k = input(f"  top_k [{sampling['top_k']}]  : ").strip()
            if k: sampling["top_k"] = int(k)
            p = input(f"  top_p [{sampling['top_p']}]  : ").strip()
            if p: sampling["top_p"] = float(p)

        except ValueError:
            print("  Invalid input — using auto values.")

    # ── Server capability probes ──────────────────────────────────────────────
    # Run llama-server --help exactly ONCE and reuse the text for all probes.
    # This avoids spawning three separate --help processes back-to-back.
    print()
    print("Probing server capabilities...")
    # Single --help call; all probes reuse this cached string.
    _srv_help = probe_server_help(SERVER_EXE)

    fa_flag    = detect_flash_attn_style(SERVER_EXE, _srv_help)
    device_fmt = detect_device_style(SERVER_EXE, _srv_help)
    # --reasoning-format controls <think> tag visibility in API responses.
    # We always pass it (not only for detected think models) because any
    # model can produce stray <think> tags if the llama.cpp build supports
    # the feature.  Passing "none" by default guarantees a clean chat UI.
    reasoning_fmt_supported = "--reasoning-format" in _srv_help

    if fa_flag is None:
        print("  Flash-attn        : not supported (skipping)")
    elif "auto" in fa_flag:
        print(f"  Flash-attn        : new style  ({fa_flag})")
    else:
        print(f"  Flash-attn        : old style  ({fa_flag})")

    if device_fmt is None:
        print("  --device          : not supported — CUDA_VISIBLE_DEVICES fallback")
    else:
        print(f"  --device          : style='{device_fmt}'  "
              f"→ '--device {device_fmt.format(idx=gpu_index)}'")

    if reasoning_fmt_supported:
        print("  --reasoning-format: supported")
    else:
        print("  --reasoning-format: not supported in this build")

    log(f"flash-attn: {fa_flag}  device_fmt: {device_fmt}  "
        f"reasoning-fmt: {reasoning_fmt_supported}")

    # Ask about <think> visibility whenever the build supports the flag.
    # If NOT supported, show_thinking value is irrelevant (flag won't be added).
    if reasoning_fmt_supported:
        show_thinking = ask_show_thinking(think_detected)
        log(f"show_thinking: {show_thinking}")

    # ── Mode selection ────────────────────────────────────────────────────────
    #
    # Four modes:
    #   (Enter / w)  web server   — llama-server + open browser  [default]
    #   c            CLI chat     — llama-cli interactive terminal
    #   r            router       — llama-server with --rpc workers
    #   k            worker       — llama-rpc-server (no model, GPU backend)

    print()
    print("  Modes:  [W]eb server (default)  |  [C]LI chat  |")
    print("          [R]outer (RPC main)      |  [K] Worker (RPC node)")
    raw_mode = input("Select mode [W/c/r/k]: ").strip().lower()

    if raw_mode in ("k", "worker"):
        # ── RPC worker mode ──────────────────────────────────────────────
        log("Mode: rpc-worker")
        run_rpc_worker(cfg, gpu_index)

    elif raw_mode in ("r", "router"):
        # ── Router (RPC main server) mode ────────────────────────────────
        # Ask for worker addresses before starting the server.
        # Workers must already be running llama-rpc-server.
        log("Mode: router")
        print()
        print("  Router mode: this server will offload layers to RPC workers.")
        print("  Workers must be running  llama-rpc-server  before you continue.")
        print("  Enter worker addresses as  host:port  separated by commas.")
        print("  Example:  192.168.1.10:50052,192.168.1.11:50052")
        rpc_input = input("  RPC workers: ").strip()
        if not rpc_input:
            print("  No workers entered — falling back to standalone server mode.")
            log("Router mode: no workers entered, running standalone")
            rpc_input = ""
        else:
            log(f"RPC workers: {rpc_input}")
        run_server(model, params, sampling, fa_flag, device_fmt,
                   show_thinking, reasoning_fmt_supported, cfg, gpu_index,
                   rpc_servers=rpc_input)

    elif raw_mode in ("c", "cli"):
        # ── CLI mode ─────────────────────────────────────────────────────
        log("Mode: cli")
        if not CLI_EXE.exists():
            sys.exit(f"[ERROR] {CLI_EXE.name} not found.")
        _cli_help = ""
        try:
            _r = subprocess.run(
                [str(CLI_EXE), "--help"],
                capture_output=True, text=True, timeout=10
            )
            _cli_help = _r.stdout + _r.stderr
        except Exception:
            pass
        run_cli(model, params, sampling, fa_flag, device_fmt,
                show_thinking, reasoning_fmt_supported, cfg, gpu_index,
                cli_help_text=_cli_help)

    else:
        # ── Web server mode (default — Enter or "w") ─────────────────────
        log("Mode: server")
        run_server(model, params, sampling, fa_flag, device_fmt,
                   show_thinking, reasoning_fmt_supported, cfg, gpu_index)

    # ── Exit summary ──────────────────────────────────────────────────────────
    print()
    print(f"  Launcher log : {LOG_FILE}")
    if SERVER_LOG.exists():
        print(f"  Server log   : {SERVER_LOG}")
    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        log(f"FATAL: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
