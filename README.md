Lightweight llama.cpp launcher (auto VRAM tuning, GPU detection, no dependencies)

Features:

automatic VRAM-aware parameter selection (ctx, batch, GPU layers)

quantisation detection from GGUF filename

multi-GPU selection

backend-aware --device detection (CUDA / Vulkan / etc.)

architecture-specific sampling defaults (Llama, Gemma, Qwen, Phi, Mistral…)

optional config.json overrides

supports both server mode and CLI chat

detects flash-attention flag style

simple logging and crash detection

It’s basically a small smart launcher for llama.cpp without needing a full web UI or heavy tooling.

If anyone finds it useful or has suggestions, I’d be happy to improve it.
