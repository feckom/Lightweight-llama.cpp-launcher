Overview

run.py is a universal launcher for llama.cpp designed to simplify running local GGUF language models. Instead of manually constructing long command lines and adjusting parameters for different llama.cpp builds, the script automates model discovery, hardware detection, parameter selection, and startup for both server and CLI workflows.

The launcher scans the models/ directory for .gguf files, lets the user select a model, reads its metadata using llama-cli --info, and automatically derives appropriate runtime parameters. It detects the model architecture (such as Gemma, Llama, Qwen, Mistral, Phi, Falcon, or MPT), extracts hints like the trained context length, and determines recommended sampling parameters if the model provides them.

The script then combines three configuration layers to build the final runtime settings. Architecture-specific defaults provide safe starting values, metadata extracted from the model can override those defaults, and an optional config.json file allows the user to explicitly override any parameter without modifying the script.

Another key feature is automatic hardware detection and GPU utilization. The launcher queries available GPU memory using nvidia-smi or Windows WMI and estimates how much VRAM the model requires based on its file size and quantization type. Using this estimate, the script determines whether the model can fully fit on the GPU, should run with partial GPU offloading, or must run on CPU. It also selects appropriate context length, batch size, and micro-batch values to avoid out-of-memory errors.

The launcher also detects reasoning (“think”) models such as DeepSeek-R1, QwQ, or Qwen thinking variants. These models emit hidden reasoning inside <think>...</think> blocks before the final answer. When such a model is detected, the user is asked whether those reasoning blocks should be visible. By default they are hidden to keep the chat output cleaner. If the installed llama.cpp build supports the --reasoning-format flag, the script automatically configures it to control whether these blocks are shown.

Compatibility with different llama.cpp builds is handled automatically. The launcher inspects the output of llama-server --help to detect supported options and determine the correct syntax for flags such as --flash-attn, --device, and --reasoning-format. This allows the script to work with both older and newer builds without modification.

The script supports multiple runtime modes. In web server mode, it launches llama-server, streams the server output to the console and a log file, waits until the /health endpoint reports readiness, and then opens the web interface in the browser automatically. In CLI mode, it runs llama-cli directly in the terminal for interactive conversation. The launcher also supports distributed RPC mode, where one machine acts as a router server and additional machines run llama-rpc-server as worker nodes providing GPU compute over the network.

During operation the launcher writes structured logs to run.log and captures the full llama.cpp server output in server.log, making debugging and configuration inspection easier.

Overall, the goal of the launcher is to provide a reliable and portable way to run local LLMs with minimal manual configuration, automatically adapting to the model, the available hardware, and the installed llama.cpp build.

Example directory structure

The launcher expects a simple directory layout. All binaries and the script are placed in the same folder, while models are stored inside the models/ directory (subfolders are allowed).

launcher/
│
├─ run.py
├─ llama-server
├─ llama-cli
├─ llama-rpc-server        (optional, required only for RPC worker mode)
│
├─ config.json             (optional)
├─ run.log
├─ server.log
│
└─ models/
    └─ your-model.gguf

The models/ directory may also contain nested folders. The launcher searches recursively, so structures like the following are also supported:

models/
├─ gemma/
│   └─ gemma-3-4b-it-Q4_K_M.gguf
├─ qwen/
│   └─ qwen2.5-7b-instruct-Q5_K_M.gguf
└─ deepseek/
    └─ deepseek-r1-Q4_K_M.gguf

When the launcher starts, it scans the entire models/ directory tree and displays all discovered GGUF models for selection.
