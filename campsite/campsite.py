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

    def direct_analyze(self, question: str, hypothesis_only: bool = False) -> dict:
        """
        Run direct analysis: hypothesis translation → artifact generation.

        Bypasses the conversation manager's clarification loop entirely.
        Requires data to be loaded first via load_data() or the records setter.

        Args:
            question: Research question to analyze
            hypothesis_only: If True, only generate the hypothesis and skip
                artifact generation (no vega_lite_spec, code, or explanation).

        Returns:
            dict: Response with hypothesis (and optionally vega_lite_spec, code, explanation)
        """
        import requests

        response = requests.post(
            url=f"http://localhost:{self._server.port}/direct_analyze",
            json={
                "question": question,
                "dataSummary": dict(self._summary_stats),
                "hypothesisOnly": hypothesis_only,
            },
        )
        result = response.json()

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n--- Hypothesis ---\n{result['hypothesis']}")
            if not hypothesis_only:
                print(f"\n--- Vega-Lite Spec ---\n{json.dumps(result['vega_lite_spec'], indent=2)}")
                print(f"\n--- Generated Code ---\n{result['code']}")
                print(f"\n--- Explanation ---\n{result['explanation']}")

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

    def check_pipeline(self, hypothesis: str) -> dict:
        """
        Run the full translation pipeline on an NL hypothesis and return failures.

        Takes a natural language hypothesis, translates it to a formal IR,
        generates a visualization artifact, and runs NL->IR and IR->VIS
        semantic invariant checks on the produced artifacts.

        Requires data to be loaded first via load_data() or the records setter.

        Args:
            hypothesis: Natural language hypothesis to evaluate

        Returns:
            dict with keys:
                ir: dict — the parsed IR AST
                code: str — generated Python visualization code
                vega_lite_spec: dict — the vega-lite specification
                failures: list[dict] — all semantic invariant violations,
                    each with invariantID, violationType, message,
                    criticality, expected, observed
        """
        import requests
        from .campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict
        from .campsite_lib.nl_extractors import NLExtractor
        from .campsite_lib.si_checkers import SICheckRunner

        # Step 1: Translate NL → formal hypothesis → visualization artifact
        response = requests.post(
            url=f"http://localhost:{self._server.port}/direct_analyze",
            json={
                "question": hypothesis,
                "dataSummary": dict(self._summary_stats),
                "hypothesisOnly": False,
            },
        )
        result = response.json()

        if "error" in result:
            raise RuntimeError(f"Pipeline failed: {result['error']}")

        formal_hypothesis = result["hypothesis"]
        vega_spec = result.get("vega_lite_spec")
        code = result.get("code", "")

        # Step 2: Parse the formal hypothesis into IR
        parse_result = parse_hypothesis(formal_hypothesis)
        ir_dict = hypothesis_to_dict(parse_result.hypothesis)

        # Collect parse errors as failures
        failures = [
            {
                "invariantID": "ir.parse",
                "violationType": "malformed",
                "message": f"IR parse error in {e.get('boundary', 'unknown')}: {e.get('message', '')}",
                "criticality": "fail",
                "expected": None,
                "observed": e.get("text", ""),
            }
            for e in (hypothesis_to_dict(err) for err in parse_result.errors)
        ]

        # Step 3: Extract NL canonical field values from the original hypothesis
        nl_extractor = NLExtractor()
        nl_values = nl_extractor.extract_all(hypothesis)

        # Step 4: Run NL->IR and IR->VIS semantic invariant checks
        runner = SICheckRunner()
        violations = runner.run_all(
            ir=ir_dict,
            nl_values=nl_values,
            artifact=vega_spec,
        )

        failures.extend(v.to_dict() for v in violations)

        return {
            "ir": ir_dict,
            "code": code,
            "vega_lite_spec": vega_spec,
            "failures": failures,
        }

    def extract_nl_invariants(self, hypotheses_df, output_dir=".", include_rationale=True):
        import csv
        import json
        import os
        from .campsite_lib.nl_extractors import NLExtractor

        COLUMNS = [
            "hypothesis_id", "hypothesis_text",
            "comparator",
            "ref.type", "ref.value",
            "qty.signature", "qty.shape", "qty.uncertainty",
            "cond.n", "conditions",
            "scope",
            "partition.structure", "partition.ordered",
            "frame",
        ]

        values_path = os.path.join(output_dir, "nl_values.csv")
        nl_extractor = NLExtractor(include_rationale=include_rationale)
        all_results = []

        with open(values_path, "w", newline="") as vf:
            val_writer = csv.writer(vf)
            val_writer.writerow(COLUMNS)

            for _, row in hypotheses_df.iterrows():
                hyp_id = row["hypothesis_id"]
                nl = row["hypothesis_text"]
                nl_values = nl_extractor.extract_all(nl)
                all_results.append((hyp_id, nl_values))

                # Comparator (raw operator symbol)
                cmp_ev = nl_values.get("comparator")
                comparator = str(cmp_ev.value) if cmp_ev and cmp_ev.exists else "ᚦ"

                # Referent sub-fields
                ref_ev = nl_values.get("referent")
                ref = ref_ev.value if ref_ev and ref_ev.exists and isinstance(ref_ev.value, dict) else {}
                ref_type = ref.get("type", "")
                ref_value = ref.get("value", "")

                # Quantity fields
                sig_ev = nl_values.get("quantity.signature")
                qty_signature = str(sig_ev.value) if sig_ev and sig_ev.exists else ""
                shp_ev = nl_values.get("quantity.shape")
                qty_shape = str(shp_ev.value) if shp_ev and shp_ev.exists else ""
                unc_ev = nl_values.get("quantity.uncertainty")
                qty_uncertainty = str(unc_ev.value) if unc_ev and unc_ev.exists else ""

                # Conditions
                cond_ev = nl_values.get("conditions")
                if cond_ev and cond_ev.exists and isinstance(cond_ev.value, list):
                    cond_n = len(cond_ev.value)
                    conditions = json.dumps(cond_ev.value)
                else:
                    cond_n = 0
                    conditions = "[]"

                # Scope
                scope_ev = nl_values.get("scope")
                scope = str(scope_ev.value) if scope_ev and scope_ev.exists else ""

                # Partition sub-fields
                part_ev = nl_values.get("partition")
                part = part_ev.value if part_ev and part_ev.exists and isinstance(part_ev.value, dict) else {}
                part_structure = part.get("structure", "")
                part_ordered = part.get("ordered", "")

                # Frame
                frame_ev = nl_values.get("frame")
                frame = str(frame_ev.value) if frame_ev and frame_ev.exists else ""

                val_writer.writerow([
                    hyp_id, nl,
                    comparator,
                    ref_type, ref_value,
                    qty_signature, qty_shape, qty_uncertainty,
                    cond_n, conditions,
                    scope,
                    part_structure, part_ordered,
                    frame,
                ])
                vf.flush()

        return all_results


