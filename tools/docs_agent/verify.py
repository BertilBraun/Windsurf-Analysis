from __future__ import annotations

import ast


class VerificationError(RuntimeError):
    pass


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    class StripDocstrings(ast.NodeTransformer):
        def _strip_body(self, node: ast.AST) -> ast.AST:
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant):
                    if isinstance(first.value.value, str):
                        node.body = body[1:]
            return node

        def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
            self.generic_visit(node)
            return self._strip_body(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
            self.generic_visit(node)
            return self._strip_body(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
            self.generic_visit(node)
            return self._strip_body(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
            self.generic_visit(node)
            return self._strip_body(node)

    return StripDocstrings().visit(tree)


def verify_python_docs_only(old_src: str, new_src: str) -> None:
    """
    Best-effort safety check: reject changes that modify Python AST beyond docstrings.
    This allows adding/updating docstrings while keeping code behavior unchanged.
    """
    try:
        old_tree = ast.parse(old_src, type_comments=True)
        new_tree = ast.parse(new_src, type_comments=True)
    except SyntaxError as e:
        raise VerificationError(f"Syntax error after doc update: {e}") from e

    old_stripped = _strip_docstrings(old_tree)
    new_stripped = _strip_docstrings(new_tree)
    if ast.dump(old_stripped, include_attributes=False) != ast.dump(new_stripped, include_attributes=False):
        raise VerificationError("Non-doc Python code changed (AST mismatch).")

