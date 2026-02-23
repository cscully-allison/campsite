"""Campsite widget - AI-powered analysis assistant for Jupyter notebooks."""

import os
import socket
import atexit
import threading
import warnings
import time
import pandas as pd
import traitlets
import anywidget
import uuid
import json

from .utils import (
    validate_and_clean_dataframe,
    extract_summary_statistics,
)


def find_free_port():
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class CampsiteServer:
    """
    Singleton Python Flask server manager.
    Ensures exactly one server thread per kernel.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.port = find_free_port()
        self._thread = None
        self._running = False
        self.logfile = os.path.join(
            os.path.dirname(__file__), "campsite_server.log"
        )

        atexit.register(self.stop)
        self._initialized = True

    def start(self):
        """Start the Flask server in a daemon thread."""
        if self._running:
            return  # already running

        # Import here to avoid circular imports
        from .campsite_server import app

        def run_server():
            """Run Flask server in thread."""
            with open(self.logfile, "a") as f:
                f.write(f"Starting Campsite server on port {self.port}\n")

            # Use werkzeug directly for more control
            from werkzeug.serving import make_server

            self._server = make_server(
                "127.0.0.1",
                self.port,
                app,
                threaded=True,
            )
            self._running = True
            self._server.serve_forever()

        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()

        # Wait for server to be ready
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: float = 5.0):
        """Wait for the server to be ready to accept connections."""
        import socket

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return  # Server is ready
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(0.05)

        raise RuntimeError(f"Campsite server failed to start within {timeout} seconds")

    def stop(self):
        """Stop the Flask server."""
        if not self._running:
            return

        with open(self.logfile, "a") as f:
            f.write(f"Shutting down Campsite server on port {self.port}\n")

        if hasattr(self, "_server"):
            self._server.shutdown()

        self._running = False
        self._thread = None


class Campsite(anywidget.AnyWidget):
    """
    Campsite widget - AI-powered analysis assistant.

    Provides an interactive chat interface for hypothesis generation
    and analysis code generation based on loaded data.
    """

    _esm = os.path.join(
        os.path.dirname(__file__), "static", "campsite.js"
    )

    _session_id = traitlets.Unicode("0").tag(sync=True)
    _node_server_endpoint = traitlets.Unicode("").tag(sync=True)
    _vis_data = traitlets.Dict({}).tag(sync=True)
    _summary_stats = traitlets.Dict({}).tag(sync=True)
    vis_configs = traitlets.Dict({}).tag(sync=True)

    suppress_warnings = False

    # Singleton server shared across all Campsite instances
    _server = CampsiteServer()

    def __init__(self):
        super().__init__()
        self._server.start()
        self._node_server_endpoint = f"http://localhost:{self._server.port}"
        self._session_id = uuid.uuid4().hex

    @property
    def records(self):
        """Get the loaded data records."""
        return self._vis_data

    @records.setter
    def records(self, df):
        """Set data records from a DataFrame."""
        self._vis_data = self.load_data(df)

    def load_data(self, in_df):
        """
        Load dataframe and extract summary statistics for visualization.

        Args:
            in_df: pandas DataFrame or data convertible to DataFrame

        Returns:
            dict: Summary statistics for the data
        """
        if not isinstance(in_df, pd.DataFrame):
            try:
                in_df = pd.DataFrame(in_df)
            except Exception:
                raise ValueError(
                    "in_df must be a pandas DataFrame or convertible to one"
                )

        if not self.suppress_warnings and in_df.empty:
            warnings.warn("load_data called with an empty DataFrame")

        o_df, report = validate_and_clean_dataframe(
            in_df, self.suppress_warnings
        )
        self._summary_stats = extract_summary_statistics(o_df)

        with open('hpc_data_summary.json', 'w+') as f:
            f.write(json.dumps(self._summary_stats))

        return self._summary_stats


    def test_server(self):
        """Test the server connection."""
        import requests

        response = requests.get(
            f"http://localhost:{self._server.port}/ping"
        )
        print(response.text)

    def test_parser(self, hyp, nl):
        """
        Test the hypothesis parser.

        Args:
            hyp: Formal hypothesis string to parse
            nl: Natural language hypothesis for comparison

        Returns:
            str: JSON response with parsed result and violations
        """
        import requests

        response = requests.post(
            url=f"http://localhost:{self._server.port}/parse",
            json={"hyp": hyp, "nl_hyp": nl}
        )
        return response.text

    def analyze(self, q):
        """
        Run analysis on a question.

        Args:
            q: Question or research hypothesis to analyze
        """
        import requests

        response = requests.post(
            url=f"http://localhost:{self._server.port}/analyze",
            json={"sessionId": self._session_id, "question": q},
        )
        self.response = response.text

    def direct_analyze(self, question: str) -> dict:
        """
        Run direct analysis: hypothesis translation → artifact generation.

        Bypasses the conversation manager's clarification loop entirely.
        Requires data to be loaded first via load_data() or the records setter.

        Args:
            question: Research question to analyze

        Returns:
            dict: Response with hypothesis, vega_lite_spec, code, explanation
        """
        import requests

        response = requests.post(
            url=f"http://localhost:{self._server.port}/direct_analyze",
            json={
                "question": question,
                "dataSummary": dict(self._summary_stats),
            },
        )
        result = response.json()

        if "error" in result:
            print(f"Error: {result['error']}")

        return result

    def test_artifact_gen(self, hypothesis, contextual_information=None):
        """
        Test the artifact generator with a hypothesis.

        Args:
            hypothesis: NL or structured hypothesis string
            contextual_information: Grammar description for structured hypotheses, or None

        Returns:
            dict: Response with code
        """
        import requests

        payload = {
            "hypothesis": hypothesis,
            "dataSummary": dict(self._summary_stats),
        }
        if contextual_information is not None:
            payload["contextual_information"] = contextual_information

        response = requests.post(
            url=f"http://localhost:{self._server.port}/generate_artifact",
            json=payload,
        )
        result = response.json()

        if "error" in result:
            print(f"Error: {result['error']}")
            return result

        print(f"--- Vega-Lite Spec ---\n{json.dumps(result['vega_lite_spec'], indent=2)}")
        print(f"\n--- Generated Code ---\n{result['code']}")
        print(f"\n--- Explanation ---\n{result['explanation']}")

        return result

    def extract_nl_invariants(self, hypotheses_df, output_dir="."):
        import csv
        import os
        from .campsite_lib.nl_extractors import NLExtractor

        CANONICAL_FIELDS = [
            "event.comparator", "event.referent",
            "event.quantity.signature", "event.quantity.conditioning",
            "event.quantity.shape", "event.quantity.uncertainty", "event.form",
            "event.referent.quantity.signature",
            "event.referent.quantity.conditioning",
            "event.referent.quantity.shape",
            "event.referent.quantity.uncertainty",
        ]

        meta_headers = ["hypothesis_id"]
        for field in CANONICAL_FIELDS:
            meta_headers += [f"{field}.value", f"{field}.confidence", f"{field}.rationale"]

        values_path = os.path.join(output_dir, "nl_values.csv")
        metadata_path = os.path.join(output_dir, "nl_metadata.csv")

        nl_extractor = NLExtractor()
        all_results = []

        with open(values_path, "w", newline="") as vf, \
             open(metadata_path, "w", newline="") as mf:
            val_writer = csv.writer(vf)
            meta_writer = csv.writer(mf)

            val_writer.writerow(["hypothesis_id"] + CANONICAL_FIELDS)
            meta_writer.writerow(meta_headers)

            for _, row in hypotheses_df.iterrows():
                hyp_id = row["hypothesis_id"]
                nl = row["hypothesis_text"]
                nl_values = nl_extractor.extract_all(nl)
                all_results.append((hyp_id, nl_values))

                val_row = [hyp_id]
                for field in CANONICAL_FIELDS:
                    ev = nl_values.get(field)
                    val_row.append(str(ev.value) if ev and ev.exists else "")
                val_writer.writerow(val_row)
                vf.flush()

                meta_row = [hyp_id]
                for field in CANONICAL_FIELDS:
                    ev = nl_values.get(field)
                    if ev and ev.exists:
                        meta_row += [str(ev.value), str(ev.confidence), ev.metadata.get("rationale", "")]
                    else:
                        meta_row += ["", "", ""]
                meta_writer.writerow(meta_row)
                mf.flush()

        return all_results


