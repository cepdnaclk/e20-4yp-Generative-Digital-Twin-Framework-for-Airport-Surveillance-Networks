# ============================================================
# Dashboard Server — serves the UI and proxies API calls
# to the remote RAG server.  Does NOT modify rag_server.py.
# Run:  python serve_dashboard.py
# Then: open http://localhost:3000
# ============================================================

import http.server
import json
import os
import urllib.request
import urllib.error

RAG_SERVER = os.environ.get("RAG_URL", "http://localhost:8001")
PORT = int(os.environ.get("DASHBOARD_PORT", "80"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

API_PATHS = ["/health", "/chat", "/ask", "/optimize", "/deploy"]


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Serves dashboard.html at '/' and proxies API calls to the RAG server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.path = "/dashboard.html"
        # Proxy API GETs (e.g. /health)
        for p in API_PATHS:
            if self.path.startswith(p):
                return self._proxy("GET")
        return super().do_GET()

    def do_POST(self):
        for p in API_PATHS:
            if self.path.startswith(p):
                return self._proxy("POST")
        self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _proxy(self, method):
        url = RAG_SERVER + self.path
        headers = {"Content-Type": "application/json"}
        data = None

        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length) if length else None

        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        print(f"  {args[0]}")


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"✅ Dashboard: http://localhost:{PORT}")
    print(f"   Proxying API → {RAG_SERVER}")
    print(f"   Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Stopped.")
