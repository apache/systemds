"""OpenAI-compatible HTTP server for local HuggingFace model inference.

Serves the /v1/completions endpoint so that SystemDS llmPredict() and
other OpenAI-compatible clients can call it directly.

Usage:
  python llm_server.py <model_name> [--port PORT]

Examples:
  python llm_server.py distilgpt2 --port 8080
  python llm_server.py Qwen/Qwen2.5-3B-Instruct --port 8080
"""

import argparse
import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class InferenceHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 512))
        temperature = float(body.get("temperature", 0.0))
        top_p = float(body.get("top_p", 0.9))

        model = self.server.model
        tokenizer = self.server.tokenizer

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
            )
        new_tokens = outputs[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        resp = {
            "choices": [{"text": text}],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(new_tokens),
            },
        }
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        sys.stderr.write("[llm_server] %s\n" % (fmt % args))


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible LLM server")
    parser.add_argument("model", help="HuggingFace model name")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}", flush=True)

    server = HTTPServer(("0.0.0.0", args.port), InferenceHandler)
    server.model = model
    server.tokenizer = tokenizer
    print(f"Serving on http://0.0.0.0:{args.port}/v1/completions", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
