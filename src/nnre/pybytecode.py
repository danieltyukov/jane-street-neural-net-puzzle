"""Decompile the tiny cloudpickled lambda that puzzle 1 uses as its forward wrapper.

``model.pt`` was saved from Python 3.10, so its code object holds CPython 3.10 bytecode (two bytes
per instruction: opcode, argument). The ``dis`` module of a newer Python cannot read it, and running
it is exactly what this project avoids, so this module carries the 3.10 numbers of the handful of
opcodes the lambda uses and evaluates them symbolically into a source expression.
"""

from __future__ import annotations

from dataclasses import dataclass

# CPython 3.10 opcode numbers (Lib/opcode.py in the 3.10 branch)
OPCODES_310 = {
    1: "POP_TOP",
    25: "BINARY_SUBSCR",
    83: "RETURN_VALUE",
    100: "LOAD_CONST",
    106: "LOAD_ATTR",
    116: "LOAD_GLOBAL",
    124: "LOAD_FAST",
    131: "CALL_FUNCTION",
    133: "BUILD_SLICE",
    160: "LOAD_METHOD",
    161: "CALL_METHOD",
}

CODE_FIELDS_310 = ("argcount", "posonlyargcount", "kwonlyargcount", "nlocals", "stacksize", "flags",
                   "code", "consts", "names", "varnames", "filename", "name", "firstlineno",
                   "linetable", "freevars", "cellvars")


@dataclass
class Instr:
    offset: int
    opname: str
    arg: int
    argrepr: str


def code_fields(args: tuple) -> dict:
    if len(args) != len(CODE_FIELDS_310):
        raise ValueError(f"expected a CPython 3.10 code object ({len(CODE_FIELDS_310)} fields), got {len(args)}")
    return dict(zip(CODE_FIELDS_310, args))


def disassemble(co: dict) -> list[Instr]:
    raw = co["code"]
    out = []
    for off in range(0, len(raw), 2):
        op, arg = raw[off], raw[off + 1]
        name = OPCODES_310.get(op)
        if name is None:
            raise ValueError(f"opcode {op} at offset {off} is not in the 3.10 table here")
        if name in ("LOAD_GLOBAL", "LOAD_METHOD", "LOAD_ATTR"):
            rep = co["names"][arg]
        elif name == "LOAD_CONST":
            rep = repr(co["consts"][arg])
        elif name == "LOAD_FAST":
            rep = co["varnames"][arg]
        else:
            rep = str(arg) if name in ("CALL_FUNCTION", "CALL_METHOD", "BUILD_SLICE") else ""
        out.append(Instr(off, name, arg, rep))
    return out


def decompile(co: dict) -> str:
    """Rebuild the source of a single-expression function by running its bytecode on strings."""
    stack: list[str] = []
    for ins in disassemble(co):
        op = ins.opname
        if op in ("LOAD_GLOBAL", "LOAD_FAST", "LOAD_CONST"):
            stack.append(ins.argrepr)
        elif op in ("LOAD_METHOD", "LOAD_ATTR"):
            stack.append(f"{stack.pop()}.{ins.argrepr}")
        elif op in ("CALL_FUNCTION", "CALL_METHOD"):
            args = [stack.pop() for _ in range(ins.arg)][::-1]
            stack.append(f"{stack.pop()}({', '.join(args)})")
        elif op == "BUILD_SLICE":
            hi, lo = stack.pop(), stack.pop()
            stack.append(f"{'' if lo == 'None' else lo}:{'' if hi == 'None' else hi}")
        elif op == "BINARY_SUBSCR":
            idx = stack.pop()
            stack.append(f"{stack.pop()}[{idx}]")
        elif op == "RETURN_VALUE":
            body = stack.pop()
            params = ", ".join(co["varnames"][:co["argcount"]])
            return f"lambda {params}: {body}"
        else:
            raise ValueError(f"cannot decompile {op}")
    raise ValueError("no RETURN_VALUE")
