import json, sys
from radon.raw import analyze
from radon.complexity import cc_visit
from radon.visitors import Function
out = []
for item in json.load(sys.stdin):
    src = item["source"]
    if isinstance(src, unicode):
        src = src.encode("utf-8")
    try:
        cc = sum(b.complexity for b in cc_visit(src) if isinstance(b, Function))
        out.append({"key": item["key"], "sloc": analyze(src).sloc, "complexity": cc})
    except SyntaxError:
        out.append({"key": item["key"], "sloc": None, "complexity": None})
json.dump(out, sys.stdout)
