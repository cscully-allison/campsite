"""Flask server for Campsite analysis backend."""

import asyncio
import json
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

from .campsite_lib.utils import AnalysisState, log
from .campsite_lib.analysis_graph import analysis_assistant, direct_analysis
from .campsite_lib.ir_parser import parse_hypothesis, hypothesis_to_dict
from .campsite_lib.si_checkers import SICheckRunner
from .campsite_lib.nl_extractors import NLExtractor


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # In-memory session store
    sessions: dict[str, AnalysisState] = {}
    query_count = [0]  # Use list for mutable counter in closure

    def run_async(coro):
        """Run an async coroutine in a synchronous context."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    @app.route("/analyze", methods=["POST"])
    def analyze():
        """Main analysis endpoint."""
        data = request.get_json()
        session_id = data.get("sessionId", "0")
        question = data.get("question", "")
        data_summary = data.get("dataSummary", {})

        # Get or create session state
        if session_id in sessions:
            state = sessions[session_id]
        else:
            # New session
            state = AnalysisState(
                dataSummary=data_summary,
                initialUserQuestion=question,
                clarifications=[],
                refinementQuestions=[],
                refinementClarifications=[],
                hypothesis=None,
                code=None,
                stage="conversation",
                clarificationNeeded=None,
                turns=0,
                clarificationTurns=0,
                waitingForUser=False,
                substage="none",
            )
            sessions[session_id] = state

        # Update follow-up questions while waiting
        if state.substage == "refinement":
            state.refinementClarifications.append(question)
        elif state.waitingForUser:
            state.clarifications.append(question)
        else:
            state.initialUserQuestion = question

        # Run the analysis graph
        result = run_async(analysis_assistant.ainvoke(state))

        # Update session with result
        if isinstance(result, dict):
            # Convert dict result back to AnalysisState
            result_state = AnalysisState(**result)
        else:
            result_state = result

        sessions[session_id] = result_state

        if result_state.waitingForUser:
            # Return prompt to frontend
            return jsonify({
                "stage": result_state.stage,
                "substage": result_state.substage,
                "waiting": True,
                "userPrompt": result_state.userPrompt,
                "hypothesis": result_state.hypothesis,
                "code": result_state.code,
            })

        if result_state.stage == "break":
            # Clear session on break
            del sessions[session_id]
            return jsonify({
                "waiting": True,
                "hypothesis": None,
                "code": None,
                "userPrompt": "Input too vague after multiple attempts to clarify. Ending analysis. Please start a new analysis with a more specific question.",
            })

        # Clear session on completion
        del sessions[session_id]

        # Return final result
        vega_spec = json.loads(result_state.vega_lite_spec) if result_state.vega_lite_spec else None
        return jsonify({
            "waiting": False,
            "hypothesis": result_state.hypothesis,
            "vega_lite_spec": vega_spec,
            "code": result_state.code,
            "explanation": result_state.explanation,
        })

    @app.route("/ping", methods=["GET"])
    def ping():
        """Health check endpoint."""
        query_count[0] += 1
        return str(query_count[0])

    @app.route("/parse", methods=["POST"])
    def parse():
        """Parse hypothesis endpoint."""
        data = request.get_json()
        hyp_text = data.get("hyp", "")
        nl_hyp = data.get("nl_hyp", "")

        # Parse the hypothesis (with partial recovery on failure)
        parse_result = parse_hypothesis(hyp_text)
        ir_dict = hypothesis_to_dict(parse_result.hypothesis)
        parse_errors = [hypothesis_to_dict(e) for e in parse_result.errors]

        # Run semantic invariant checks
        runner = SICheckRunner()
        if nl_hyp:
            nl_extractor = NLExtractor()
            nl_values = nl_extractor.extract_all(nl_hyp)
            violations = runner.run_all(nl_values=nl_values, ir=ir_dict)
        else:
            violations = runner.run_ir_only(ir=ir_dict)

        all_violations = [v.to_dict() for v in violations]

        return jsonify({
            "parsed": ir_dict,
            "parseErrors": parse_errors,
            "violations": all_violations,
        })
    
    
    @app.route("/direct_analyze", methods=["POST"])
    def direct_analyze():
        """Direct analysis: hypothesis → artifact with no clarification loop."""
        data = request.get_json()
        question = data.get("question", "")
        data_summary = data.get("dataSummary", {})
        hypothesis_only = data.get("hypothesisOnly", False)

        try:
            result_state = run_async(direct_analysis(question, data_summary, hypothesis_only=hypothesis_only))
            vega_spec = (
                json.loads(result_state.vega_lite_spec)
                if result_state.vega_lite_spec
                else None
            )
            return jsonify({
                "hypothesis": result_state.hypothesis,
                "vega_lite_spec": vega_spec,
                "code": result_state.code,
                "explanation": result_state.explanation,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/generate_artifact", methods=["POST"])
    def generate_artifact_endpoint():
        """Generate a visualization artifact from a hypothesis."""
        from .campsite_lib.artifact_gen import generate_artifact

        data = request.get_json()
        hypothesis = data.get("hypothesis", "")
        data_summary = data.get("dataSummary", {})
        contextual_information = data.get("contextual_information", None)

        try:
            result, code_result = run_async(generate_artifact(
                hypothesis=hypothesis,
                data_summary=data_summary,
                contextual_information=contextual_information,
            ))
            return jsonify({
                "vega_lite_spec": result.parsed_vega_lite_spec,
                "code": code_result,
                "explanation": result.explanation,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# Create the app instance
app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 3000):
    """Run the Flask server."""
    print(f"READY:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    run_server(port=port)
