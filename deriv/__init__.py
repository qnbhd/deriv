# MIT License
#
# Copyright (c) 2023 Templin Konstantin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ast
import io
import sys
from typing import TYPE_CHECKING, Annotated, cast

__all__ = [
    "Printer",
    "Deriver",
    "Simplifier",
    "take_derivative",
    "take_derivative_tree",
]

Op2Sym = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.MatMult: "*",
    ast.Div: "/",
    ast.Pow: "^",
}
UnOp2Sym = {ast.USub: "-", ast.UAdd: "+", **Op2Sym}  # type: ignore


class Visitor(ast.NodeVisitor):
    def visit(self, node: ast.AST | ast.expr | ast.stmt):
        return super().visit(cast(ast.AST, node))


class Printer(Visitor):
    def __init__(self, stream=sys.stdout):
        self.stream = stream

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        assert isinstance(node.op, ast.unaryop | ast.operator)
        self.print(UnOp2Sym[node.op.__class__])  # noqa
        self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if self._need_parens(node.op, node.left):
            self.print("(")
            self.visit(node.left)
            self.print(")")
        else:
            self.visit(node.left)
        self.print(f" {Op2Sym[node.op.__class__]} ")  # noqa
        if self._need_parens(node.op, node.right):
            self.print("(")
            self.visit(node.right)
            self.print(")")
        else:
            self.visit(node.right)

    def visit_Constant(self, node: ast.Constant) -> None:
        self.print(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        self.print(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        assert isinstance(node.func, ast.Name)
        self.print(node.func.id)
        self.print("(")
        for i, arg in enumerate(node.args):
            self.visit(arg)
            if i < len(node.args) - 1:
                self.print(", ")
        self.print(")")

    def print(self, value, *args, **kwargs):
        print(value, *args, sep="", end="", file=self.stream, **kwargs)

    @staticmethod
    def _need_parens(op, node):
        return isinstance(op, ast.Div | ast.Mult) and not isinstance(
            node, ast.Constant | ast.Name | ast.Call | ast.UnaryOp
        )


class Deriver(Visitor):
    def __init__(self, by):
        self.by = by

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        return self.visit(node.value)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        return self.visit(node.body[0])

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp:
        return ast.UnaryOp(op=node.op, operand=self.visit(node.operand))

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST | ast.BinOp:
        x = self.visit(node.left)
        y = self.visit(node.right)
        if TYPE_CHECKING:
            xs: ast.AST | ast.Call | ast.BinOp | ast.UnaryOp | None = None
        match node.op:
            # dX(x + y) => dX(x) + dX(y)
            case ast.Add() | ast.Sub():
                return ast.BinOp(left=x, right=y, op=node.op)
            # dX(x*y) = dX(x)y + xdX(y)
            case ast.Mult():
                xs = ast.BinOp(left=x, right=node.right, op=ast.Mult())
                ys = ast.BinOp(left=node.left, right=y, op=ast.Mult())
                return ast.BinOp(left=xs, right=ys, op=ast.Add())
            # dX(x/y) = ( dX(x)y - xdX(y) ) / y^2
            case ast.Div():
                xs = ast.BinOp(left=x, right=node.right, op=ast.Mult())
                ys = ast.BinOp(left=node.left, right=y, op=ast.Mult())
                zs = ast.BinOp(
                    left=node.right, right=ast.Constant(value=2), op=ast.Pow()
                )
                xyw = ast.BinOp(left=xs, right=ys, op=ast.Sub())
                return ast.BinOp(left=xyw, right=zs, op=ast.Div())
            # dX(x^n) = f^g(dX(g)*ln(f) + g*dX(f)/f)
            case ast.Pow():
                # g'*ln(f)
                xs = ast.Call(func=ast.Name(id="ln"), args=[node.left])
                ys = ast.BinOp(left=y, right=xs, op=ast.Mult())
                # g*f'/f
                zs = ast.BinOp(left=x, right=node.left, op=ast.Div())
                ws = ast.BinOp(left=node.right, right=zs, op=ast.Mult())
                # g'*ln(f) + g*f'/f
                us = ast.BinOp(left=ys, right=ws, op=ast.Add())
                # f^g(g'*ln(f) + g*f'/f)
                rs = ast.BinOp(left=node.left, right=node.right, op=node.op)
                return ast.BinOp(left=rs, right=us, op=ast.Mult())
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return ast.Constant(value=0)

    def visit_Name(self, node: ast.Name) -> ast.Constant:
        if node.id == self.by:
            return ast.Constant(value=1)
        return ast.Constant(0)

    def visit_Call(self, node: ast.Call) -> ast.BinOp:
        assert isinstance(node.func, ast.Name)
        # call with one arg only supported
        assert len(node.args) == 1
        arg, *_ = node.args
        if TYPE_CHECKING:
            result: ast.AST | ast.Call | ast.BinOp | ast.UnaryOp | None = None
        match node.func.id:
            case "sin":
                result = ast.Call(func=ast.Name(id="cos"), args=node.args)
            case "cos":
                result = ast.Call(func=ast.Name(id="sin"), args=node.args)
                result = ast.UnaryOp(op=ast.USub(), operand=result)
            case "ln":
                result = ast.BinOp(left=ast.Constant(value=1), right=arg, op=ast.Div())
            case "exp":
                result = ast.Call(func=node.func, args=node.args)
            case "sqrt":
                xs = ast.Call(func=ast.Name(id="sqrt"), args=node.args)
                ys = ast.BinOp(left=ast.Constant(value=2), right=xs, op=ast.Mult())
                result = ast.BinOp(left=ast.Constant(value=1), right=ys, op=ast.Div())
            case _:
                raise RuntimeError(f"Unknown func name {_}")
        return ast.BinOp(left=result, right=self.visit(arg), op=ast.Mult())


class Simplifier(Visitor):
    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        return self.visit(node.value)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        return self.visit(node.body[0])

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST | ast.expr:
        match node.op:
            case ast.UAdd():
                return node.operand
        return node

    def visit_BinOp(
        self, node: ast.BinOp
    ) -> ast.AST | ast.Constant | ast.UnaryOp | ast.BinOp:
        xs = self.visit(node.left)
        ys = self.visit(node.right)
        base = ast.BinOp(left=xs, right=ys, op=node.op)
        match node.op:
            case ast.Add() | ast.Sub():
                # c1 + c2 => c3 := c1 + c2
                operator = (
                    (lambda x, y: x + y)
                    if isinstance(node.op, ast.Add)
                    else (lambda x, y: x - y)
                )
                if isinstance(xs, ast.Constant) and isinstance(ys, ast.Constant):
                    return ast.Constant(value=operator(xs.value, ys.value))
                if self._is_zero(xs):
                    # 0 + c1 => c1
                    if isinstance(node.op, ast.Add):
                        return ys
                    # 0 - c1 => -c1
                    if isinstance(node.op, ast.Sub):
                        return ast.UnaryOp(op=ast.Sub(), operand=ys)
                # c1 + 0 => c1
                # c1 - 0 => c1
                if self._is_zero(ys):
                    return xs
            case ast.Mult() | ast.Div():
                operator = (
                    (lambda x, y: x * y)
                    if isinstance(node.op, ast.Mult)
                    else lambda x, y: x / y
                )
                # c1 * c2 => c3 := c1 * c2
                if isinstance(xs, ast.Constant) and isinstance(ys, ast.Constant):
                    return ast.Constant(value=operator(xs.value, ys.value))
                if self._is_zero(xs) or self._is_zero(ys):
                    return ast.Constant(value=0)
                # x / 1 => x
                # x * 1 => x
                if self._is_one(ys):
                    return xs
                # 1 * x => x
                if self._is_one(xs) and isinstance(node.op, ast.Mult):
                    return ys
        return base

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        assert isinstance(node.func, ast.Name)
        # call with one arg only supported
        assert len(node.args) == 1
        arg, *_ = node.args
        return ast.Call(func=node.func, args=[self.visit(arg)])

    @staticmethod
    def _is_zero(node) -> bool:
        return isinstance(node, ast.Constant) and node.value == 0

    @staticmethod
    def _is_one(node) -> bool:
        return isinstance(node, ast.Constant) and node.value == 1


def take_derivative_tree(
    expr: Annotated[str, "expression"] | ast.AST, by: Annotated[str, "with respect to"]
) -> ast.AST:
    """
    Takes the derivative of a given expression represented as either a string
    or an Abstract Syntax Tree (AST) object with respect to a specified variable.

    Args:
        expr: The expression to differentiate, either as a string
            or an AST object.
        by: The variable with respect to which the derivative is computed.

    Returns:
        The AST representing the derivative of the input expression.

    Note:
        '^' will be replaced with "**" if u using string as `expr` value type
    """
    tree = ast.parse(expr.replace("^", "**")) if isinstance(expr, str) else expr
    derivative = Deriver(by).visit(tree)
    return Simplifier().visit(derivative)


def take_derivative(
    expr: Annotated[str, "expression"] | ast.AST, by: Annotated[str, "with respect to"]
) -> Annotated[str, "derivative"]:
    """
    Takes the derivative of a given expression represented as either a string
    or an Abstract Syntax Tree (AST) object with respect to a specified variable
    and returns the result as a string.

    Args:
        expr: The expression to differentiate, either as a string
            or an AST object.
        by: The variable with respect to which the derivative is computed.

    Returns:
        The string representation of the derivative of the input expression.
    """
    string_io = io.StringIO()
    Printer(string_io).visit(take_derivative_tree(expr, by))
    return string_io.getvalue()
