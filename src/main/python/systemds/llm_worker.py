"""
SystemDS LLM Worker - Python side of the Py4J bridge.
Java starts this script, then calls generate() via Py4J.
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters

class LLMWorker:
    def __init__(self, model_name="distilgpt2"):
        print(f"Loading model: {model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        print(f"Model loaded: {model_name}", flush=True)

    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=float(temperature) > 0.0
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    class Java:
        implements = ["org.apache.sysds.api.jmlc.LLMCallback"]

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "distilgpt2"
    java_port = int(sys.argv[2]) if len(sys.argv) > 2 else 25333

    print(f"Starting LLM worker, connecting to Java on port {java_port}", flush=True)
    
    worker = LLMWorker(model_name)

    # Connect to Java's GatewayServer and register this worker
    # The callback_server starts a server on Python's side for Java to call back
    # Use port 25334 which Java's CallbackClient expects
    gateway = JavaGateway(
        gateway_parameters=GatewayParameters(port=java_port),
        callback_server_parameters=CallbackServerParameters(port=25334)
    )
    
    print(f"Python callback server started on port 25334", flush=True)
    
    gateway.entry_point.registerWorker(worker)
    print("Worker registered with Java, waiting for requests...", flush=True)
    
    # Keep the worker alive to handle callbacks from Java
    # The callback server runs in a daemon thread, so we need to block here
    import threading
    shutdown_event = threading.Event()
    try:
        # Wait indefinitely until Java closes the connection or kills the process
        shutdown_event.wait()
    except KeyboardInterrupt:
        print("Worker shutting down", flush=True)