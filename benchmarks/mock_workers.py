"""
Lightweight Mock Model Workers (Ports 8001 & 8002)
Simulates GPU KV-Cache prefill latency and streaming generation:
- Cold Cache (Unseen Prefix): 160ms - 180ms prefill latency
- Warm Cache (Cached Prefix): 18ms - 24ms prefill latency (TTFT drop!)
"""

import http.server
import json
import socketserver
import sys
import threading
import time

class ModelWorkerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging in terminal
        return

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}

            messages = payload.get("messages", [])
            stream = payload.get("stream", False)

            # Extract system prompt prefix
            system_prompt = ""
            for m in messages:
                if m.get("role") == "system":
                    system_prompt = m.get("content", "")
                    break

            worker_cache = self.server.kv_cache
            port = self.server.server_address[1]

            # Simulate GPU KV Cache Prefill Latency
            is_cache_hit = system_prompt and (system_prompt in worker_cache)

            if is_cache_hit:
                # Warm Cache: KV tensors already loaded in GPU memory!
                prefill_delay = 0.020  # ~20ms
                cache_status = "WARM HIT"
            else:
                # Cold Cache: GPU must compute full attention matrix from scratch
                prefill_delay = 0.175  # ~175ms
                cache_status = "COLD PREFILL"
                if system_prompt:
                    worker_cache.add(system_prompt)

            time.sleep(prefill_delay)

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                # Stream 5 simulated tokens
                tokens = [" [Worker", f"-{str(port)[-1]}]", " Generated", " response", " chunk."]
                for tok in tokens:
                    chunk = {
                        "choices": [
                            {"delta": {"content": tok}, "index": 0, "finish_reason": None}
                        ]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.005)

                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                resp = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": payload.get("model", "qwen2.5:1.5b"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"[Worker-{port[-1]}|{cache_status}] Calculation complete.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

def run_worker(port):
    server = ThreadedHTTPServer(("127.0.0.1", port), ModelWorkerHandler)
    server.kv_cache = set()
    print(f"  [+] Worker online on http://127.0.0.1:{port} (Simulated vLLM/Ollama GPU Node)")
    server.serve_forever()

def start_both_workers():
    print("Starting 2 downstream model workers...")
    t1 = threading.Thread(target=run_worker, args=(8001,), daemon=True)
    t2 = threading.Thread(target=run_worker, args=(8002,), daemon=True)
    t1.start()
    t2.start()
    time.sleep(1)
    print("Both workers active. Ready for traffic.")

if __name__ == "__main__":
    start_both_workers()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nWorkers stopped.")
