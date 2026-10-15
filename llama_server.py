"""Ciclo de vida do llama-server: um processo por arquivo de modelo."""

import subprocess
import time

import requests


class ServerError(RuntimeError):
    pass


class LlamaServer:
    def __init__(self, binary, model_path, host, port, args, log_path, load_timeout_s=300):
        self.url = f"http://{host}:{port}"
        self.cmd = [binary, "-m", str(model_path), "--host", host, "--port", str(port), *args]
        self.log_path = log_path
        self.load_timeout_s = load_timeout_s
        self.proc = None
        self.load_time_s = None

    @property
    def pid(self):
        return self.proc.pid

    def start(self):
        if self._healthy():
            raise ServerError(f"já existe um servidor respondendo em {self.url}; encerre-o antes")
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT)

    def wait_ready(self):
        """Bloqueia até /health responder 200; o tempo decorrido é o tempo de carga."""
        deadline = self._t0 + self.load_timeout_s
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise ServerError(f"llama-server encerrou (código {self.proc.returncode}):\n{self.log_tail()}")
            if self._healthy():
                self.load_time_s = round(time.perf_counter() - self._t0, 3)
                return
            time.sleep(0.1)
        raise ServerError(f"timeout de {self.load_timeout_s}s aguardando o modelo carregar")

    def _healthy(self):
        try:
            return requests.get(f"{self.url}/health", timeout=1).status_code == 200
        except requests.RequestException:
            return False

    def model_meta(self):
        data = requests.get(f"{self.url}/v1/models", timeout=10).json()
        return (data.get("data") or [{}])[0].get("meta", {})

    def chat(self, messages, gen, timeout):
        payload = {
            "messages": messages,
            "temperature": gen["temperature"],
            "top_k": gen["top_k"],
            "seed": gen["seed"],
            "max_tokens": gen["max_tokens"],
            "stream": False,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        r = requests.post(f"{self.url}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n=30):
        self._log.flush()
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.proc:
            self._log.close()
