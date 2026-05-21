"""IR Parser using Lark - converts hypothesis strings to AST."""

import re

from lark import Lark, Transformer, v_args, UnexpectedInput
from .ir_ast import (
    Hypothesis,
    Comparison,
    Unspecified,
    RV,
    Expectation,
    Contrast,
    Predicate,
    Const,
    AttrVar,
    ConstVar,
    BinaryExpr,
    FuncCall,
    ConditionedExpr,
    Extract,
    Partition,
    PartitionPred,
    CrossPartition,
    ErrorNode,
    ParseResult,
    VALID_COMPARATORS,
)

# Lark grammar based on grammar_v3_revised.txt specification
GRAMMAR = r"""
    ?start: hypothesis

    // Hypothesis: event with optional ACROSS/WITHIN partition clauses
    hypothesis: event "ACROSS" partition "WITHIN" partition -> hyp_across_within
              | event "ACROSS" partition -> hyp_across
              | event "WITHIN" partition -> hyp_within
              | event -> hyp_plain

    // Partition: attr with optional pred, or cross product via X operator
    // Chain pattern avoids left recursion: partition_atom ("X" partition_atom)*
    partition: partition_atom ("X" partition_atom)* -> partition_chain

    partition_atom: IDENTIFIER partition_pred? -> partition_node

    partition_pred: comparator_or_error pred_ref -> partition_pred_node

    // Event: quantity comparator referent (no event-level predicate)
    event: quantity comparator_or_error referent -> event_node
         | quantity THORN referent -> event_unspecified_comparator
         | quantity comparator_or_error THORN -> event_unspecified_referent

    // Referent: const or func(var, ...)
    referent: const_val -> referent_const
            | FUNC_NAME "(" var_list ")" -> referent_func
            | IDENTIFIER "(" var_list ")" -> referent_func

    var_list: var ("," var)* -> var_list_node

    // Quantity of interest (V3 = maximal)
    ?quantity: rv
             | estimand
             | expr

    // Random variable: Distribution(estimand) or Distribution(expr)
    rv: IDENTIFIER "(" estimand ")" -> rv_estimand_node
      | IDENTIFIER "(" expr ")" -> rv_expr_node

    // Estimands
    ?estimand: contrast
             | expectation

    // Expectation: E[expr] — no pipe separator
    expectation: "E" "[" expr "]" -> expectation_node

    // Contrast: expectation fop expectation
    contrast: expectation fop expectation -> contrast_node

    // Predicates with conjunction/disjunction support (^ and v operators)
    predicate: predicate_atom (LOP predicate_atom)* -> predicate_compound
    LOP: "^" | "v"

    predicate_atom: IDENTIFIER comparator_or_error pred_ref -> predicate_comparison

    // pred-ref: const | extract | pred-expr (no wildcard)
    ?pred_ref: pred_expr

    // pred-expr: chain pattern to avoid left recursion
    pred_expr: pred_atom (fop pred_atom)* -> pred_expr_chain

    pred_atom: extract
             | FUNC_NAME "(" pred_expr_list? ")" -> pred_func_node
             | IDENTIFIER "(" pred_expr_list? ")" -> pred_func_node
             | const_value
             | IDENTIFIER -> pred_attr_var

    pred_expr_list: pred_expr ("," pred_expr)* -> pred_expr_list_node

    // Expressions: chain of expr_atoms connected by fop operators
    // This allows binary ops between conditioned expressions:
    // salary (dept = "eng") - salary (dept = "sci")
    expr: expr_atom (fop expr_atom)* -> expr_chain

    // Expression atoms: subject with optional parenthesized predicate
    expr_atom: subject_chain predicate_clause -> conditioned_expr
             | subject_chain -> plain_expr

    // Predicate clause: predicate wrapped in literal parentheses
    predicate_clause: "(" predicate ")"

    // Subject: chain pattern for binary expressions within atoms
    subject_chain: subject_atom (fop subject_atom)* -> subject_chain_node

    // Subject atoms, including optional grouping parentheses
    // paren_group wraps expr (not just subject_chain) so conditioned
    // sub-expressions like (salary (dept = "eng")) work inside parens
    ?subject_atom: "(" expr ")" -> paren_group
                 | extract
                 | FUNC_NAME "(" expr_list? ")" -> func_node
                 | IDENTIFIER "(" expr_list? ")" -> func_node
                 | var

    // Model extraction: Extract(model, expr)
    extract.2: "Extract" "(" IDENTIFIER "," expr ")" -> extract_node

    expr_list: expr ("," expr)* -> expr_list_node

    var: IDENTIFIER -> attr_var
       | const_value -> const_var

    const_val: const_value -> const_node

    // Constants: number, string, tuple, or boolean
    const_value: BOOLEAN -> boolean
              | NUMBER -> number
              | STRING -> string
              | "(" NUMBER "," NUMBER ")" -> tuple_val

    BOOLEAN: "true" | "false"

    comparator_or_error: COMPARATOR -> comparator

    COMPARATOR: ">=" | "<=" | "!=" | ">" | "<" | "=" | "BETWEEN" | "IN"

    THORN: "\u16A6"

    fop: FOP
    FOP: "+" | "-" | "*" | "/"

    // Standard aggregate/statistical functions
    FUNC_NAME.2: "AVG" | "MAX" | "MIN" | "SUM" | "COUNT"
               | "MEDIAN" | "VARIANCE" | "STDDEV"
               | "CORR" | "PERCENTILE" | "QUANTILE"

    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER: /[0-9]+(\.[0-9]+)?/
    STRING: /"[^"]*"/

    %import common.WS
    %ignore WS
"""


@v_args(inline=True)
class HypothesisTransformer(Transformer):
    """Transform Lark parse tree to AST dataclasses."""

    # -- Hypothesis level --

    def hyp_plain(self, event):
        return Hypothesis(type="hypothesis", event=event)

    def hyp_across(self, event, partition):
        return Hypothesis(type="hypothesis", event=event, across_partition=partition)

    def hyp_within(self, event, partition):
        return Hypothesis(type="hypothesis", event=event, within_partition=partition)

    def hyp_across_within(self, event, across_part, within_part):
        return Hypothesis(
            type="hypothesis",
            event=event,
            across_partition=across_part,
            within_partition=within_part,
        )

    # -- Partitions --

    def partition_chain(self, *items):
        """Left-fold [partition_atom, partition_atom, ...] into nested CrossPartition."""
        if len(items) == 1:
            return items[0]
        result = items[0]
        for i in range(1, len(items)):
            result = CrossPartition(type="cross_partition", lhs=result, rhs=items[i])
        return result

    def partition_node(self, attr, *rest):
        pred = rest[0] if rest else None
        return Partition(type="partition", attr=str(attr), pred=pred)

    def partition_pred_node(self, comparator, value):
        return PartitionPred(type="partition_pred", comparator=comparator, value=value)

    # -- Events --

    def event_node(self, quantity, comparator, referent):
        """Event: quantity comparator referent."""
        return Comparison(
            type="comparison",
            quantity=quantity,
            comparator=comparator,
            referent=referent,
        )

    def event_unspecified_comparator(self, quantity, _thorn, referent):
        """Event: quantity ᚦ referent (comparator unspecified)."""
        return Comparison(
            type="comparison",
            quantity=quantity,
            comparator="ᚦ",
            referent=referent,
        )

    def event_unspecified_referent(self, quantity, comparator, _thorn):
        """Event: quantity comparator ᚦ (referent unspecified)."""
        return Comparison(
            type="comparison",
            quantity=quantity,
            comparator=comparator,
            referent=Unspecified(),
        )

    # -- Referents --

    def referent_const(self, val):
        return val

    def referent_func(self, name, var_list):
        return FuncCall(type="func", name=str(name), args=var_list)

    def var_list_node(self, *vars):
        return list(vars)

    # -- Quantity pass-through --

    def quantity(self, item):
        return item

    def estimand(self, item):
        return item

    # -- Expressions --

    def expr_chain(self, *items):
        """Left-fold [expr_atom, op, expr_atom, ...] into nested BinaryExpr."""
        if len(items) == 1:
            return items[0]
        result = items[0]
        for i in range(1, len(items), 2):
            result = BinaryExpr(
                type="binary", lhs=result, op=str(items[i]), rhs=items[i + 1]
            )
        return result

    def conditioned_expr(self, subject, predicate):
        return ConditionedExpr(
            type="conditioned_expr", expr=subject, predicate=predicate
        )

    def plain_expr(self, item):
        return item

    def predicate_clause(self, predicate):
        return predicate

    def subject_chain_node(self, *items):
        """Left-fold [atom, op, atom, ...] into nested BinaryExpr."""
        if len(items) == 1:
            return items[0]
        result = items[0]
        for i in range(1, len(items), 2):
            result = BinaryExpr(
                type="binary", lhs=result, op=str(items[i]), rhs=items[i + 1]
            )
        return result

    def paren_group(self, inner):
        """Pass-through for optional grouping parentheses around subject."""
        return inner

    def subject_atom(self, item):
        return item

    # -- RV --

    def rv_estimand_node(self, dist, estimand):
        return RV(type="rv", distribution=str(dist), estimand=estimand)

    def rv_expr_node(self, dist, expr):
        return RV(type="rv", distribution=str(dist), estimand=expr)

    # -- Estimands --

    def expectation_node(self, expr):
        return Expectation(type="expectation", expr=expr)

    def contrast_node(self, left, op, right):
        return Contrast(type="contrast", lhs=left, op=str(op), rhs=right)

    # -- Predicates --

    def predicate_compound(self, *items):
        """Handle predicate with ^ (conjunction) and v (disjunction) operators."""
        if len(items) == 1:
            return items[0]
        # items = [pred, lop, pred, lop, pred, ...]
        result = items[0]
        for i in range(1, len(items), 2):
            op = str(items[i])
            kind = "conjunction" if op == "^" else "disjunction"
            result = Predicate(
                type="predicate",
                kind=kind,
                lhs=result,
                rhs=items[i + 1],
            )
        return result

    def predicate_comparison(self, attr, comparator, value):
        return Predicate(
            type="predicate",
            kind="comparison",
            attr=str(attr),
            comparator=comparator,
            value=value,
        )

    def pred_ref(self, item):
        return item

    def pred_atom(self, item):
        return item

    def pred_attr_var(self, name):
        return AttrVar(type="attr", name=str(name))

    def pred_expr_chain(self, *items):
        """Left-fold [atom, op, atom, ...] into nested BinaryExpr."""
        if len(items) == 1:
            return items[0]
        result = items[0]
        for i in range(1, len(items), 2):
            result = BinaryExpr(type="binary", lhs=result, op=str(items[i]), rhs=items[i + 1])
        return result

    def pred_func_node(self, name, *args):
        arg_list = args[0] if args else []
        return FuncCall(type="func", name=str(name), args=arg_list)

    def pred_expr_list_node(self, *exprs):
        return list(exprs)

    # -- Extract --

    def extract_node(self, model, expr):
        """Handle Extract(model, expr) expressions."""
        return Extract(
            type="extract",
            model=str(model),
            expr=expr,
        )

    # -- Function calls and expressions --

    def func_node(self, name, *args):
        arg_list = args[0] if args else []
        return FuncCall(type="func", name=str(name), args=arg_list)

    def expr_list_node(self, *exprs):
        return list(exprs)

    # -- Variables and constants --

    def attr_var(self, name):
        return AttrVar(type="attr", name=str(name))

    def const_var(self, value):
        return ConstVar(type="const", value=value)

    def const_node(self, value):
        return Const(type="const", value=value)

    def boolean(self, b):
        return str(b) == "true"

    def number(self, n):
        return float(n)

    def string(self, s):
        # Remove quotes from string
        s_str = str(s)
        if s_str.startswith('"') and s_str.endswith('"'):
            return s_str[1:-1]
        return s_str

    def tuple_val(self, a, b):
        return (float(a), float(b))

    def comparator(self, c):
        return str(c)

    def fop(self, op):
        return str(op)

    def FOP(self, token):
        return str(token)

    def IDENTIFIER(self, token):
        return str(token)

    def NUMBER(self, token):
        return float(token)

    def STRING(self, token):
        return str(token)


# Create parser instance
_parser = None


def get_parser():
    """Get or create the Lark parser instance."""
    global _parser
    if _parser is None:
        _parser = Lark(GRAMMAR, start="start", parser="earley", ambiguity="resolve")
    return _parser


def parse_hypothesis(text: str) -> ParseResult:
    """
    Parse a hypothesis string into an AST with partial recovery on failure.

    Args:
        text: The hypothesis string to parse

    Returns:
        ParseResult containing the Hypothesis AST and any errors encountered.
        When errors is empty, the full parse succeeded.
        When errors is non-empty, partial recovery was attempted.
    """
    parser = get_parser()
    transformer = HypothesisTransformer()

    try:
        tree = parser.parse(text)
        hypothesis = transformer.transform(tree)
        return ParseResult(hypothesis=hypothesis, errors=[])
    except UnexpectedInput as e:
        # Full parse failed — attempt partial recovery
        return _attempt_recovery(text, e)
    except Exception as e:
        # Unexpected error — no recovery possible
        err = ErrorNode(
            type="error",
            boundary="event",
            message=f"Parse error: {str(e)}",
            text=text,
        )
        return ParseResult(
            hypothesis=Hypothesis(type="hypothesis", event=err),
            errors=[err],
        )


# ---------------------------------------------------------------------------
# Partial recovery helpers
# ---------------------------------------------------------------------------

# Comparators ordered longest-first for greedy matching
_COMPARATOR_PATTERNS = [
    (">=", re.compile(r">=")),
    ("<=", re.compile(r"<=")),
    ("!=", re.compile(r"!=")),
    ("BETWEEN", re.compile(r"\bBETWEEN\b")),
    ("IN", re.compile(r"\bIN\b")),
    (">", re.compile(r">")),
    ("<", re.compile(r"<")),
    ("=", re.compile(r"=")),
    ("\u16A6", re.compile(r"\u16A6")),
]


def _find_event_comparator(text: str):
    """Find the event-level comparator at bracket/paren depth 0.

    Returns (comparator_str, start_index, end_index) or None.
    Scans left-to-right, longest-match-first. Returns the first match at depth 0.
    """
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "([":
            depth += 1
            i += 1
            continue
        if ch in ")]":
            depth = max(0, depth - 1)
            i += 1
            continue

        if depth == 0:
            for comp_str, pattern in _COMPARATOR_PATTERNS:
                m = pattern.match(text, i)
                if m:
                    return (comp_str, m.start(), m.end())

        i += 1

    return None


def _try_parse_segment(text: str, role: str):
    """Try to parse a text segment as a quantity or referent using dummy wrappers.

    For quantity: wraps as '{text} > 0' and extracts .event.quantity.
    For referent: wraps as 'x > {text}' and extracts .event.referent.

    Returns:
        (node, error_or_none)
    """
    parser = get_parser()
    transformer = HypothesisTransformer()
    text_stripped = text.strip()

    if not text_stripped:
        err = ErrorNode(
            type="error",
            boundary=role,
            message=f"Empty {role}",
            text=text,
        )
        return err, err

    if role == "quantity":
        wrapper = f"{text_stripped} > 0"
        try:
            tree = parser.parse(wrapper)
            result = transformer.transform(tree)
            return result.event.quantity, None
        except Exception as e:
            err = ErrorNode(
                type="error",
                boundary="quantity",
                message=f"Parse error in quantity: {_first_line(e)}",
                text=text_stripped,
            )
            return err, err

    elif role == "referent":
        wrapper = f"x > {text_stripped}"
        try:
            tree = parser.parse(wrapper)
            result = transformer.transform(tree)
            return result.event.referent, None
        except Exception as e:
            err = ErrorNode(
                type="error",
                boundary="referent",
                message=f"Parse error in referent: {_first_line(e)}",
                text=text_stripped,
            )
            return err, err

    # Unknown role — shouldn't happen
    err = ErrorNode(type="error", boundary="event", message=f"Unknown role: {role}", text=text_stripped)
    return err, err


def _first_line(e: Exception) -> str:
    """Extract the first line of an exception message."""
    return str(e).split("\n")[0]


def _attempt_recovery(text: str, original_error: Exception) -> ParseResult:
    """Attempt partial recovery when full parsing fails.

    Strategy:
      1. Find the event-level comparator at bracket depth 0
      2. Split input into quantity_text and rest_text
      3. Parse each segment independently via dummy wrappers
      4. Return Comparison with ErrorNode for failed segments
    """
    errors = []

    comp_match = _find_event_comparator(text)
    if comp_match is None:
        # No comparator found — can't recover any structure
        err = ErrorNode(
            type="error",
            boundary="event",
            message=f"Parse error: {_first_line(original_error)}",
            text=text,
            start=getattr(original_error, "pos_in_stream", 0) or 0,
            end=len(text),
        )
        return ParseResult(
            hypothesis=Hypothesis(type="hypothesis", event=err),
            errors=[err],
        )

    comp_str, comp_start, comp_end = comp_match
    quantity_text = text[:comp_start]
    rest_text = text[comp_end:]

    # Parse quantity segment
    quantity_node, q_err = _try_parse_segment(quantity_text, "quantity")
    if q_err:
        q_err.start = 0
        q_err.end = comp_start
        errors.append(q_err)

    # Parse referent segment
    referent_node, r_err = _try_parse_segment(rest_text, "referent")
    if r_err:
        r_err.start = comp_end
        r_err.end = len(text)
        errors.append(r_err)

    # Validate comparator
    if comp_str not in VALID_COMPARATORS:
        comp_err = ErrorNode(
            type="error",
            boundary="comparator",
            message=f"Invalid comparator: {comp_str}",
            text=comp_str,
            start=comp_start,
            end=comp_end,
        )
        errors.append(comp_err)

    comparison = Comparison(
        type="comparison",
        quantity=quantity_node,
        comparator=comp_str,
        referent=referent_node,
    )

    return ParseResult(
        hypothesis=Hypothesis(type="hypothesis", event=comparison),
        errors=errors,
    )


def hypothesis_to_dict(h) -> dict:
    """Convert a hypothesis AST to a dictionary for JSON serialization."""
    if hasattr(h, "__dataclass_fields__"):
        result = {}
        for field_name in h.__dataclass_fields__:
            value = getattr(h, field_name)
            result[field_name] = hypothesis_to_dict(value)
        return result
    elif isinstance(h, list):
        return [hypothesis_to_dict(item) for item in h]
    elif isinstance(h, tuple):
        return list(h)
    else:
        return h


def parse_result_to_dict(pr: ParseResult) -> dict:
    """Convert a ParseResult to a dictionary for JSON serialization."""
    return {
        "hypothesis": hypothesis_to_dict(pr.hypothesis),
        "errors": [hypothesis_to_dict(e) for e in pr.errors],
    }
