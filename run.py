#!/usr/bin/env python3
"""
run.py  —  launcher for llama.cpp  (server + CLI mode)

Changelog vs previous version:
  1. Quantisation detection uses regex instead of substring:
       r'[_\-.](iq[2-4]|q[2-8])(?:_k(?:_[smlx]+)?|_[0-9]+)?(?:[_.\-]|$)'
     Prevents false matches like "q4" inside "q4something_weird".
  2. --device flag style is probed from --help and adapted per backend:
       numeric  →  --device 2
       colon    →  --device cuda:2  /  Vulkan:2
       name     →  --device CUDA2   /  Vulkan2
       absent   →  flag skipped, CUDA_VISIBLE_DEVICES used as fallback
  3. ARCH_SAMPLING_DEFAULTS: phi sets temp=0.0 for deterministic mode;
     table now covers phi-3 / phi-4 variants explicitly.
  4. Note: KV cache VRAM estimate (point 3 from review) intentionally kept
     as 1.15× overhead — accurate per-layer math would require a full GGUF
     metadata parser and is out of scope for a dependency-free launcher.

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

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent.resolve()
MODEL_DIR   = SCRIPT_DIR / "models"
SERVER_EXE  = SCRIPT_DIR / "llama-server.exe"
CLI_EXE     = SCRIPT_DIR / "llama-cli.exe"
LOG_FILE    = SCRIPT_DIR / "run.log"
SERVER_LOG  = SCRIPT_DIR / "server.log"
CONFIG_FILE = SCRIPT_DIR / "config.json"

DEFAULTS = {
    "threads":       6,
    "threads_batch": 12,
    "host":          "127.0.0.1",
    "port":          8080,
    "ready_timeout": 120,
}

ARCH_TAGS = ["gemma", "llama", "mistral", "phi", "qwen", "falcon", "mpt"]

# ── Improvement 3 / point 4: per-architecture sampling defaults ───────────────
#
# phi / phi-3 / phi-4: temp=0.0  →  deterministic (greedy) decoding.
#   Microsoft's own inference examples use temperature=0 for Phi models.
# gemma: temp=1.0, top_k=64 as per Google's recommended settings.
# qwen: top_k=20, top_p=0.8 per Alibaba's documented defaults.
# All others: conservative general-purpose values.
#
# config.json keys  sampling_temp / sampling_top_k / etc.  override these.

ARCH_SAMPLING_DEFAULTS = {
    "gemma":    dict(temp=1.0,  top_k=64,  top_p=0.95, repeat_penalty=1.0,  chat_template="gemma"),
    "llama":    dict(temp=0.6,  top_k=40,  top_p=0.9,  repeat_penalty=1.1,  chat_template="chatml"),
    "mistral":  dict(temp=0.7,  top_k=50,  top_p=0.9,  repeat_penalty=1.05, chat_template="mistral"),
    "phi":      dict(temp=0.0,  top_k=50,  top_p=0.95, repeat_penalty=1.0,  chat_template="chatml"),
    "qwen":     dict(temp=0.7,  top_k=20,  top_p=0.8,  repeat_penalty=1.05, chat_template="chatml"),
    "falcon":   dict(temp=0.8,  top_k=40,  top_p=0.9,  repeat_penalty=1.1,  chat_template="chatml"),
    "mpt":      dict(temp=0.7,  top_k=40,  top_p=0.9,  repeat_penalty=1.1,  chat_template="chatml"),
    "_default": dict(temp=0.7,  top_k=40,  top_p=0.9,  repeat_penalty=1.1,  chat_template="chatml"),
}

def get_sampling_defaults(arch: str, model_info: str, cfg: dict) -> dict:
    """
    Three-layer sampling config (lowest → highest priority):
      1. ARCH_SAMPLING_DEFAULTS table
      2. Values parsed from model metadata (--info output)
      3. config.json  sampling_<key>  or flat  <key>  entries
    """
    base = dict(ARCH_SAMPLING_DEFAULTS.get(arch, ARCH_SAMPLING_DEFAULTS["_default"]))
    base.update(parse_sampling_from_info(model_info))
    for key in ("temp", "top_k", "top_p", "repeat_penalty", "chat_template"):
        for cfg_key in (f"sampling_{key}", key):
            if cfg_key in cfg:
                base[key] = cfg[cfg_key]
                break
    return base

def parse_sampling_from_info(info_text: str) -> dict:
    """Extract sampling hints from llama-cli --info output via regex."""
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
                result[key] = float(val) if "." in val else int(val)
            except ValueError:
                pass
    return result

# ── Logging ────────────────────────────────────────────────────────────────────

_log_lock = threading.Lock()

def init_logs():
    """Truncate both run.log and server.log at startup."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("===== llama.cpp launcher log =====\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    open(SERVER_LOG, "w").close()

def log(msg: str, also_print: bool = True):
    line = msg.rstrip()
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    if also_print:
        print(line)

# ── config.json ───────────────────────────────────────────────────────────────

def load_config() -> dict:
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

# ── Model info (single subprocess call) ───────────────────────────────────────

def get_model_info(model_path: Path) -> str:
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
    Priority: explicit key=value line → full-text scan → binary header → filename.
    Regex on key=value line prevents false positives from build metadata.
    """
    if info_text:
        kv = re.compile(r"(?:general\.architecture|arch)\s*[=:]\s*(\w+)", re.I)
        for line in info_text.splitlines():
            m = kv.search(line)
            if m:
                val = m.group(1).lower()
                for arch in ARCH_TAGS:
                    if arch in val:
                        return arch
        low = info_text.lower()
        for arch in ARCH_TAGS:
            if arch in low:
                return arch
    try:
        with open(model_path, "rb") as f:
            data = f.read(8192)
        text = data.decode("utf-8", errors="ignore").lower()
        for arch in ARCH_TAGS:
            if arch in text:
                return arch
    except Exception as e:
        log(f"  [WARN] Header read failed: {e}", also_print=False)
    name = model_path.name.lower()
    for arch in ARCH_TAGS:
        if arch in name:
            return arch
    return "unknown"

def parse_ctx_hint(info_text: str) -> int:
    if not info_text:
        return 0
    m = re.search(r"(?:context[_-]length|n_ctx_train)\s*[=:]\s*(\d+)", info_text, re.I)
    if m:
        val = int(m.group(1))
        if 512 <= val <= 131072:
            return val
    return 0

# ── VRAM detection ─────────────────────────────────────────────────────────────

def get_vram_via_nvidia_smi() -> tuple:
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
        pass
    except Exception as e:
        log(f"  [WARN] nvidia-smi failed: {e}", also_print=False)
    return 0, []

def get_vram_via_wmi() -> int:
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
    vram_mb, gpus = get_vram_via_nvidia_smi()
    if vram_mb > 0:
        return vram_mb, gpus
    vram_mb = get_vram_via_wmi()
    if vram_mb > 0:
        return vram_mb, [{"index": 0, "name": "GPU (WMI)",
                          "total_mb": vram_mb, "free_mb": vram_mb}]
    return 0, []

def select_gpu(gpus: list) -> int:
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
        print(f"  [WARN] Index {choice} not found — using GPU {gpus[0]['index']}.")
    except ValueError:
        print(f"  [WARN] Invalid input — using GPU {gpus[0]['index']}.")
    return gpus[0]["index"]

# ── Improvement 1: precise quantisation regex ─────────────────────────────────
#
# Pattern breakdown:
#   [_\-.]          — quantisation tag must follow a separator (not mid-word)
#   (iq[2-4]|q[2-8]) — named group: IQ2/IQ3/IQ4 or Q2–Q8
#   (?:_k(?:_[smlx]+)?|_[0-9]+)?  — optional K-quant suffix (_K_M, _K_S, _0 …)
#   (?:[_.\-]|$)    — must be followed by separator or end of string
#
# This prevents "q4something_weird" from matching as Q4 while correctly
# matching "Q4_K_M", "Q4_0", "Q5_K_XL", "IQ4_XS", etc.

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
    Estimate GPU VRAM needed for full model offload.
    Uses file_size × quant_ratio × 1.15 overhead factor.
    The 1.15× covers activation buffers and fixed CUDA allocations.
    KV cache scales with ctx size and is handled separately by the ctx tier.
    """
    file_mb = model_path.stat().st_size / (1024 * 1024)
    name    = model_path.name

    ratio = 1.05  # default: F16 / unknown
    m = _QUANT_RE.search(name)
    if m:
        key = m.group("q").lower()
        # normalise iq* → q* for ratio lookup
        ratio_key = key if key in QUANT_VRAM_RATIO else key.replace("iq", "q")
        ratio = QUANT_VRAM_RATIO.get(ratio_key, 1.05)

    return int(file_mb * ratio * 1.15)

def auto_params(vram_free_mb: int, has_gpu: bool, model_path: Path) -> dict:
    if not has_gpu:
        return dict(ctx=2048, ngl=0, batch=512, ubatch=512)

    needed_mb = estimate_vram_needed_mb(model_path)
    ratio     = vram_free_mb / needed_mb if needed_mb > 0 else 0

    if   ratio >= 1.0:  ngl = 99
    elif ratio >= 0.5:  ngl = 50
    elif ratio >= 0.25: ngl = 20
    else:               ngl = 0

    if   vram_free_mb >= 12000: ctx, batch, ubatch = 16384, 2048, 2048
    elif vram_free_mb >= 8000:  ctx, batch, ubatch = 8192,  1024, 1024
    elif vram_free_mb >= 4000:  ctx, batch, ubatch = 4096,  512,  512
    else:                       ctx, batch, ubatch = 2048,  512,  512

    return dict(ctx=ctx, ngl=ngl, batch=batch, ubatch=ubatch)

# ── Improvement 2: --device style detection ───────────────────────────────────
#
# llama.cpp --device syntax varies by backend version:
#
#   numeric  →  --device 0          (older CUDA builds)
#   name     →  --device CUDA0      (current CUDA / Vulkan by name)
#   colon    →  --device cuda:0     (some builds, both CUDA and Vulkan)
#
# We probe the actual --help output and format the device string accordingly.
# If --device is absent entirely, we fall back to CUDA_VISIBLE_DEVICES only.

_DEVICE_PATTERNS = [
    # Most specific patterns first
    (re.compile(r'--device\s+cuda:\d',   re.I), "cuda:{idx}"),
    (re.compile(r'--device\s+vulkan:\d', re.I), "Vulkan:{idx}"),
    (re.compile(r'--device\s+CUDA\d'        ), "CUDA{idx}"),
    (re.compile(r'--device\s+Vulkan\d'      ), "Vulkan{idx}"),
    (re.compile(r'--device\s+<int>',     re.I), "{idx}"),
    # Generic fallback: --device present but style unclear → use CUDA name form
    (re.compile(r'--device',             re.I), "CUDA{idx}"),
]

def detect_device_style(server_exe: Path) -> str | None:
    """
    Returns a format string like "CUDA{idx}" or "cuda:{idx}" or "{idx}",
    or None if the build does not support --device at all.
    Callers use  fmt.format(idx=gpu_index)  to build the actual value.
    """
    try:
        result = subprocess.run(
            [str(server_exe), "--help"],
            capture_output=True, text=True, timeout=10
        )
        help_text = result.stdout + result.stderr
        if "--device" not in help_text.lower():
            return None
        for pattern, fmt in _DEVICE_PATTERNS:
            if pattern.search(help_text):
                return fmt
    except Exception:
        pass
    return None

def build_device_args(device_fmt: str | None, gpu_index: int) -> list:
    """
    Return ["--device", "<value>"] or [] if device flag not supported.
    Only appended when a non-default GPU is selected (index != 0) OR
    when the build uses a non-numeric format that requires explicit naming.
    """
    if device_fmt is None:
        return []
    # Numeric style: index 0 is the default, skip flag to reduce noise
    if device_fmt == "{idx}" and gpu_index == 0:
        return []
    return ["--device", device_fmt.format(idx=gpu_index)]

# ── flash-attn detection ───────────────────────────────────────────────────────

def detect_flash_attn_style(server_exe: Path):
    """
    Returns:
        None                  — not supported
        "--flash-attn"        — old boolean style
        "--flash-attn auto"   — new value style (on|off|auto required)
    """
    try:
        result = subprocess.run(
            [str(server_exe), "--help"],
            capture_output=True, text=True, timeout=10
        )
        help_text = result.stdout + result.stderr
        if "--flash-attn" not in help_text and "-fa" not in help_text:
            return None
        for line in help_text.splitlines():
            if "--flash-attn" in line or "-fa," in line:
                if "on|off|auto" in line or "[on" in line:
                    return "--flash-attn auto"
                else:
                    return "--flash-attn"
        return "--flash-attn auto"
    except Exception:
        return None

# ── Port check ─────────────────────────────────────────────────────────────────

def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

# ── Output streaming ───────────────────────────────────────────────────────────

def stream_output(stream, log_path: Path, stop_event: threading.Event):
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

# ── Health check ───────────────────────────────────────────────────────────────

def wait_for_health(host: str, port: int, timeout: int,
                    proc: subprocess.Popen,
                    stop_event: threading.Event) -> bool:
    url = f"http://{host}:{port}/health"
    for i in range(1, timeout + 1):
        if stop_event.is_set():
            raise KeyboardInterrupt
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
            pass
        pct = int(i * 100 / timeout)
        bar = ("=" * int(i * 30 / timeout)).ljust(30)
        print(f"  Loading [{bar}] {pct}%   ", end="\r", flush=True)
        time.sleep(1)
    print()
    return False

# ── Server mode ────────────────────────────────────────────────────────────────

def run_server(model: Path, params: dict, sampling: dict,
               fa_flag, device_fmt, cfg: dict, gpu_index: int):

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
    cmd.extend(build_device_args(device_fmt, gpu_index))
    if fa_flag:
        cmd.extend(fa_flag.split())

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
    proc = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, cwd=str(SCRIPT_DIR), env=env,
        )
        log(f"PID: {proc.pid}", also_print=False)

        reader = threading.Thread(
            target=stream_output,
            args=(proc.stdout, SERVER_LOG, stop_event),
            daemon=True
        )
        reader.start()

        try:
            ready = wait_for_health(host, port, cfg["ready_timeout"],
                                    proc, stop_event)
        except KeyboardInterrupt:
            raise

        if not ready:
            if proc.poll() is None:
                print()
                log("[WARNING] Server not responding within timeout.")
                print(f"  Check {SERVER_LOG.name} for error details.")
                if input("Open browser anyway? (y/n): ").strip().lower() != "y":
                    log("Launch cancelled.")
                    stop_event.set()
                    proc.terminate()
                    return
            else:
                return
        else:
            time.sleep(0.5)

        url = f"http://{host}:{port}"
        log(f"Opening browser: {url}")
        os.startfile(url)
        print()
        print(f"  Server : {url}")
        print(f"  Logs   : {LOG_FILE.name}  |  {SERVER_LOG.name}")
        print("  Ctrl+C to stop.")
        print()

        proc.wait()
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
                proc.kill()
        log("Server stopped.")

# ── CLI mode ───────────────────────────────────────────────────────────────────

def run_cli(model: Path, params: dict, sampling: dict,
            fa_flag, device_fmt, cfg: dict, gpu_index: int):

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
        "--interactive",
        "--chat-template",  sampling["chat_template"],
    ]
    cmd.extend(build_device_args(device_fmt, gpu_index))
    if fa_flag:
        cmd.extend(fa_flag.split())

    log(f"Command: {' '.join(cmd)}", also_print=False)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    print()
    print("-" * 70)
    print("  CLI chat  |  Ctrl+C to exit")
    print("-" * 70)
    print()

    try:
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env)
        log(f"CLI exited with code {result.returncode}.")
        if result.returncode != 0:
            print(f"\n[ERROR] llama-cli exited with code {result.returncode}.")
    except KeyboardInterrupt:
        log("CLI interrupted.")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_logs()
    cfg = load_config()

    if not SERVER_EXE.exists():
        sys.exit(f"[ERROR] {SERVER_EXE.name} not found in {SCRIPT_DIR}")
    if not MODEL_DIR.exists():
        sys.exit(f"[ERROR] models/ directory not found in {SCRIPT_DIR}")
    if not CLI_EXE.exists():
        print(f"[WARNING] {CLI_EXE.name} not found — "
              "CLI mode and architecture detection unavailable.")
        log(f"WARNING: {CLI_EXE.name} missing")

    models = sorted(MODEL_DIR.glob("*.gguf"))
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

    model    = models[choice - 1]
    size_gb  = model.stat().st_size / 1_073_741_824
    print()
    print(f"Selected  : {model.name}  (~{size_gb:.1f} GB)")
    log(f"Selected: {model}  (~{size_gb:.1f} GB)")

    # Single --info call; arch, ctx, and sampling all parsed from same output
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

    params    = auto_params(vram_free, has_gpu, model)
    needed_mb = estimate_vram_needed_mb(model)
    fit_pct   = int(vram_free / needed_mb * 100) if needed_mb and has_gpu else 0

    for key in ("ctx", "ngl", "batch", "ubatch"):
        if key in cfg:
            params[key] = cfg[key]

    if ctx_max > 0 and params["ctx"] > ctx_max:
        print(f"  Capping ctx {params['ctx']} → {ctx_max} (model maximum)")
        log(f"ctx capped to {ctx_max}")
        params["ctx"] = ctx_max

    sampling = get_sampling_defaults(arch, model_info, cfg)

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
    print()
    log(f"ctx={params['ctx']} ngl={params['ngl']} batch={params['batch']} fit={fit_pct}%")
    log(f"sampling: temp={sampling['temp']} top_k={sampling['top_k']} "
        f"top_p={sampling['top_p']} template={sampling['chat_template']}")

    confirm = input("Use these parameters? (Enter=yes / n=manual): ").strip().lower()
    if confirm == "n":
        try:
            params["ctx"]    = int(input("  ctx tokens          : "))
            params["ngl"]    = int(input("  GPU layers (0=CPU)  : "))
            params["batch"]  = int(input("  batch size          : "))
            params["ubatch"] = int(input("  ubatch size         : "))
            t = input(f"  temp [{sampling['temp']}]  : ").strip()
            if t: sampling["temp"] = float(t)
            k = input(f"  top_k [{sampling['top_k']}]  : ").strip()
            if k: sampling["top_k"] = int(k)
            p = input(f"  top_p [{sampling['top_p']}]  : ").strip()
            if p: sampling["top_p"] = float(p)
        except ValueError:
            print("  Invalid input — using auto values.")

    print()
    print("Probing server capabilities...")
    fa_flag    = detect_flash_attn_style(SERVER_EXE)
    device_fmt = detect_device_style(SERVER_EXE)

    if fa_flag is None:
        print("  Flash-attn  : not supported (skipping)")
    elif "auto" in fa_flag:
        print(f"  Flash-attn  : new style  ({fa_flag})")
    else:
        print(f"  Flash-attn  : old style  ({fa_flag})")

    if device_fmt is None:
        print("  --device    : not supported — using CUDA_VISIBLE_DEVICES fallback")
    else:
        ex = device_fmt.format(idx=gpu_index)
        print(f"  --device    : style='{device_fmt}'  → will pass '--device {ex}'")

    log(f"Flash-attn: {fa_flag}  device_fmt: {device_fmt}")

    print()
    mode = input("Open web chat interface? (y/n): ").strip().lower()
    log(f"Mode: {'server' if mode == 'y' else 'cli'}")

    if mode == "y":
        run_server(model, params, sampling, fa_flag, device_fmt, cfg, gpu_index)
    else:
        if not CLI_EXE.exists():
            sys.exit(f"[ERROR] {CLI_EXE.name} not found.")
        run_cli(model, params, sampling, fa_flag, device_fmt, cfg, gpu_index)

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
