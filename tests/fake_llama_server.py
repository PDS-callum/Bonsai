"""A stand-in for llama-server: answers /health, chat and raw completions.

Takes the same --port option (ignoring the rest). Every chat reply echoes the
last line of the prompt and claims the answer is 42, so tests can check what
was asked and what came back. A raw completion of token ids continues by
counting up from the last one. With --fail, it exits at once instead.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, body: dict[str, object]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        self._send(200, {"status": "ok"})

    def do_POST(self) -> None:
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if request.get("repeat_penalty") != 1.0:
            self._send(400, {"error": "requests must turn off the repeat penalty"})
            return
        if self.path == "/completion":
            last = request["prompt"][-1]
            count = request["n_predict"]
            self._send(200, {"tokens": list(range(last + 1, last + 1 + count))})
            return
        prompt = request["messages"][-1]["content"]
        text = f"{prompt.splitlines()[-1]}\nThe answer is: 42"
        self._send(
            200,
            {
                "choices": [{"message": {"content": text}}],
                "usage": {"completion_tokens": 7},
            },
        )


def main() -> None:
    if "--fail" in sys.argv:
        sys.exit(5)
    port = int(sys.argv[sys.argv.index("--port") + 1])
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
