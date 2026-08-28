import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

request_count = 0

class MockAzureSidecar(BaseHTTPRequestHandler):
    def do_GET(self):
        global request_count
        if self.path == "/ping":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
        elif self.path == "/metrics":
            request_count += 1
            
            # Simulate a metric pattern
            # CPU spike every 10 requests, otherwise normal
            if request_count % 10 == 0:
                cpu = 95.0
                log = "CRITICAL: CPU SPIKE detected! High computation thread started."
            elif request_count % 8 == 0:
                cpu = 15.2
                log = "INFO: GC overhead limit stable. Garbage collection cleaned 120MB."
            elif request_count % 12 == 0:
                cpu = 20.1
                log = "ERROR: OUT OF MEMORY - heap space exhausted."
            else:
                cpu = 10.0 + (request_count % 5) * 2.5
                log = f"INFO: Service processing request number {request_count}."
                
            # Simulate a slow memory leak
            ram = 30.0 + (request_count * 0.5) % 50.0
            heap = 100.0 + (request_count * 2.0) % 200.0
            
            data = {
                "cpu_percent": cpu,
                "memory_percent": ram,
                "memory_used_mb": heap,
                "gc_count": request_count // 8,
                "log_message": log,
                "language": "python"
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run(port=8080):
    server = HTTPServer(("localhost", port), MockAzureSidecar)
    print(f"Mock Azure Sidecar listening on http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Stopping server...")

if __name__ == "__main__":
    import sys
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run(port)
