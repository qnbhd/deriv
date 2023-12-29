from deriv import take_derivative


def test_simple():
    assert take_derivative("c", "x") == "0"
    assert take_derivative("x", "x") == "1"


def test_binary():
    assert take_derivative("x + y", "x") == "1"
    assert take_derivative("x + y", "y") == "1"
    assert take_derivative("x - y", "x") == "1"
    assert take_derivative("x - y", "y") == "-1"
    assert take_derivative("x * y", "x") == "y"
    assert take_derivative("x * y", "y") == "x"
    assert take_derivative("x / y", "x") == "y / (y ^ 2)"
    assert take_derivative("y / x", "x") == "-y / (x ^ 2)"
    assert take_derivative("x ^ y", "x") == "(x ^ y) * (y * (1 / x))"
    assert take_derivative("y ^ x", "x") == "(y ^ x) * ln(y)"


def test_functions():
    assert take_derivative("sin(x)", "x") == "cos(x)"
    assert take_derivative("cos(x)", "x") == "-sin(x)"
    assert take_derivative("ln(x)", "x") == "1 / x"
    assert take_derivative("exp(x)", "x") == "exp(x)"
    assert take_derivative("sqrt(x)", "x") == "1 / (2 * sqrt(x))"


def test_complex():
    assert take_derivative("cos(sin(x))", "x") == "-sin(sin(x)) * cos(x)"
    assert take_derivative("ln(x+y+(x*y))", "x") == "(1 / (x + y + x * y)) * (1 + y)"
    assert take_derivative("sin(2^x)", "x") == "cos(2 ^ x) * ((2 ^ x) * ln(2))"


def test_repeats():
    expr = "sin(x)"
    expr = take_derivative(expr, "x")
    assert expr == "cos(x)"
    expr = take_derivative(expr, "x")
    assert expr == "-sin(x)"
