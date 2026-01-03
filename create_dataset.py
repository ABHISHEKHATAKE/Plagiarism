import ast
import networkx as nx
from collections import Counter
from math import sqrt
import tokenize
import io
import keyword

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0

def bin_flow(x):
    if x < 0.3:
        return 0
    elif x < 0.6:
        return 1
    else:
        return 2


#AST
class ASTAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.counts = Counter()
        self.total = 0
        self.depth = 0
        self.max_depth = 0
        self.identifiers = set()

    def generic_visit(self, node):
        self.total += 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        super().generic_visit(node)
        self.depth -= 1

    def visit_If(self, n): self.counts["If"] += 1; self.generic_visit(n)
    def visit_For(self, n): self.counts["Loop"] += 1; self.generic_visit(n)
    def visit_While(self, n): self.counts["Loop"] += 1; self.generic_visit(n)
    def visit_Assign(self, n): self.counts["Assign"] += 1; self.generic_visit(n)
    def visit_AugAssign(self, n): self.counts["Assign"] += 1; self.generic_visit(n)
    def visit_Return(self, n): self.counts["Return"] += 1; self.generic_visit(n)
    def visit_Compare(self, n): self.counts["Compare"] += 1; self.generic_visit(n)
    def visit_BinOp(self, n): self.counts["BinOp"] += 1; self.generic_visit(n)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load)) and node.id not in keyword.kwlist and node.id not in dir(__builtins__):
            self.identifiers.add(node.id)
        self.generic_visit(node)

def get_ast_features(code):
    t = ast.parse(code)
    a = ASTAnalyzer()
    a.visit(t)
    total = a.total or 1

    return {
        "If_ratio": a.counts["If"] / total,
        "Loop_ratio": a.counts["Loop"] / total,
        "Assign_ratio": a.counts["Assign"] / total,
        "Return_ratio": a.counts["Return"] / total,
        "Compare_ratio": a.counts["Compare"] / total,
        "AST_Depth": a.max_depth,
        "AST_Diversity": len(a.counts) / total,
        "Unique_Identifiers": len(a.identifiers)
    }


#CFG
class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.g = nx.DiGraph()
        self.i = 0
        self.entry = self.new()
        self.exit = self.new()
        self.cur = self.entry

    def new(self):
        n = self.i
        self.g.add_node(n)
        self.i += 1
        return n

    def e(self, u, v):
        self.g.add_edge(u, v)

    def simple(self):
        n = self.new()
        self.e(self.cur, n)
        self.cur = n

    def visit_Assign(self, n): self.simple()
    def visit_AugAssign(self, n): self.simple()
    def visit_Expr(self, n): self.simple()

    def visit_If(self, n):
        c = self.new()
        self.e(self.cur, c)
        a = self.new()
        self.cur = c
        for s in n.body: self.visit(s)
        self.e(self.cur, a)
        self.e(c, a)
        self.cur = a

    def visit_For(self, n): self.visit_While(n)

    def visit_While(self, n):
        c = self.new()
        self.e(self.cur, c)
        self.cur = c
        for s in n.body: self.visit(s)
        self.e(self.cur, c)
        a = self.new()
        self.e(c, a)
        self.cur = a

    def visit_Return(self, n):
        r = self.new()
        self.e(self.cur, r)
        self.e(r, self.exit)
        self.cur = self.exit

def get_cfg_features(code):
    t = ast.parse(code)
    b = CFGBuilder()
    b.visit(t)
    if b.cur != b.exit:
        b.e(b.cur, b.exit)

    N = b.g.number_of_nodes()
    E = b.g.number_of_edges()
    cyclo = E - N + 2

    return {
        "CFG_Nodes": N,
        "CFG_Edges": E,
        "Cyclomatic": cyclo,
        "Branch_Density": cyclo / N if N else 0
    }


#flow path
class FlowCFG(ast.NodeVisitor):
    def __init__(self):
        self.g = nx.DiGraph()
        self.lbl = {}
        self.i = 0
        self.entry = self.new("ENTRY")
        self.exit = self.new("EXIT")
        self.cur = self.entry

    def new(self, label):
        n = self.i
        self.g.add_node(n)
        self.lbl[n] = label
        self.i += 1
        return n

    def e(self, u, v):
        self.g.add_edge(u, v)

    def simple(self):
        if self.lbl[self.cur] == "ENTRY":
            return
        n = self.new("Stmt")
        self.e(self.cur, n)
        self.cur = n

    def visit_Assign(self, n): self.simple()
    def visit_AugAssign(self, n): self.simple()
    def visit_Expr(self, n): self.simple()

    def visit_For(self, n): self.loop(n)
    def visit_While(self, n): self.loop(n)

    def loop(self, n):
        c = self.new("Loop")
        self.e(self.cur, c)
        self.cur = c
        for s in n.body: self.visit(s)
        self.e(self.cur, c)
        a = self.new("AfterLoop")
        self.e(c, a)
        self.cur = a

    def visit_If(self, n):
        c = self.new("If")
        self.e(self.cur, c)
        a = self.new("AfterIf")
        self.cur = c
        for s in n.body: self.visit(s)
        self.e(self.cur, a)
        self.e(c, a)
        self.cur = a

    def visit_Return(self, n):
        r = self.new("Return")
        self.e(self.cur, r)
        self.e(r, self.exit)
        self.cur = self.exit

def flow_similarity(code1, code2):
    def paths(code):
        t = ast.parse(code)
        b = FlowCFG()
        b.visit(t)
        if b.cur != b.exit:
            b.e(b.cur, b.exit)
        res = set()
        def dfs(n, p):
            if len(p) == 3 or b.g.out_degree(n) == 0:
                res.add(tuple(p))
                return
            for x in b.g.successors(n):
                dfs(x, p + [b.lbl[x]])
        dfs(b.entry, ["ENTRY"])
        return res

    return jaccard(paths(code1), paths(code2))


# ============================== #
# NEW FEATURE EXTRACTION HELPERS #
# ============================== #

def get_loc(code):
    return len([line for line in code.split('\n') if line.strip()])


COMMON_KEYWORDS = ['def', 'class', 'for', 'while', 'if', 'elif', 'else', 'return', 'import', 'from', 'as', 'try', 'except', 'finally', 'with', 'break', 'continue', 'pass', 'lambda', 'global', 'nonlocal', 'yield', 'async', 'await']
COMMON_OPERATORS = ['+', '-', '*', '/', '%', '**', '//', '=', '+=', '-=', '*=', '/=', '%=', '**=', '//=', '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not', 'in', 'is', '|', '&', '^', '~', '<<', '>>']

def get_token_features(code):
    tokens = []
    total_tokens = 0
    unique_keywords = set()
    unique_operators = set()

    try:
        for toknum, tokval, (srow, scol), (erow, ecol), line in tokenize.generate_tokens(io.StringIO(code).readline):
            if tokval.strip(): # Exclude empty tokens
                total_tokens += 1
                if tokval in COMMON_KEYWORDS:
                    unique_keywords.add(tokval)
                elif tokval in COMMON_OPERATORS:
                    unique_operators.add(tokval)
                tokens.append(tokval)
    except tokenize.TokenError:
        # Handle cases where code might be malformed
        pass

    # Calculate ratios based on total_tokens, avoid division by zero
    total_tokens_safe = total_tokens if total_tokens > 0 else 1

    keyword_ratio = len(unique_keywords) / total_tokens_safe
    operator_ratio = len(unique_operators) / total_tokens_safe

    return {
        "Keyword_Ratio": keyword_ratio,
        "Operator_Ratio": operator_ratio,
        "Unique_Keyword_Count": len(unique_keywords),
        "Unique_Operator_Count": len(unique_operators)
    }



# ============================== #
# FINAL FEATURE EXTRACTOR        #
# ============================== #
def extract_features(code1, code2):
    ast1 = get_ast_features(code1)
    ast2 = get_ast_features(code2)
    cfg1 = get_cfg_features(code1)
    cfg2 = get_cfg_features(code2)

    loc1 = get_loc(code1)
    loc2 = get_loc(code2)

    token_feats1 = get_token_features(code1)
    token_feats2 = get_token_features(code2)

    flow_sim = flow_similarity(code1, code2)

    features = {}

    # Add AST features and their differences
    for k in ast1:
        features[f"A_{k}"] = ast1[k]
        features[f"B_{k}"] = ast2[k]

    # Add CFG features and their differences
    for k in cfg1:
        features[f"A_{k}"] = cfg1[k]
        features[f"B_{k}"] = cfg2[k]

    # Add LOC features and their differences
    features["A_LOC"] = loc1
    features["B_LOC"] = loc2

    # Add Token features (Keyword/Operator Ratios and Counts) and their differences
    for k in token_feats1:
        features[f"A_{k}"] = token_feats1[k]
        features[f"B_{k}"] = token_feats2[k]


    # Calculate differences for existing and new features
    features["AST_Depth_Diff"] = abs(ast1["AST_Depth"] - ast2["AST_Depth"])
    features["CFG_Node_Diff"] = abs(cfg1["CFG_Nodes"] - cfg2["CFG_Nodes"])
    features["LOC_Diff"] = abs(loc1 - loc2)
    features["Unique_Identifiers_Diff"] = abs(ast1["Unique_Identifiers"] - ast2["Unique_Identifiers"])
    features["Keyword_Ratio_Diff"] = abs(token_feats1["Keyword_Ratio"] - token_feats2["Keyword_Ratio"])
    features["Operator_Ratio_Diff"] = abs(token_feats1["Operator_Ratio"] - token_feats2["Operator_Ratio"])
    features["Unique_Keyword_Count_Diff"] = abs(token_feats1["Unique_Keyword_Count"] - token_feats2["Unique_Keyword_Count"])
    features["Unique_Operator_Count_Diff"] = abs(token_feats1["Unique_Operator_Count"] - token_feats2["Unique_Operator_Count"])

    features["Flow_Path_Bin"] = bin_flow(flow_sim)

    return features

import csv

# =====================================================
# 1️⃣ DEFINE YOUR DATA (THREE LIST STRATEGY)
# =====================================================

code_list_1 = [
    """
def g(y):
    i = 0
    while i < 30:
        print(i)
        i += 1
""",

    """
def h(a):
    for i in range(30):
        print(i)
""",

    """
def k(x):
    if x > 0:
        return x
""",
    """
def add(a,b):
    return a+b
""",
"""

def f(n):
    for i in range(n):
        print(i)
""",
"""
def count(lst):
    for x in lst:
        print(x)
""",
"""def max_val(a, b):
    if a > b:
        return a
    return b
""",
"""
def f(n):
    for i in range(n):
        print(i)
"""
,
"""
def sum_vals(a, b):
    return a + b
""",
"""
def fact(n):
    if n == 0:
        return 1
    return n * fact(n-1)
""",
"""
def print_even(n):
    for i in range(n):
        if i % 2 == 0:
            print(i)
""",
"""
def sum_n(n):
    s = 0
    for i in range(n):
        s += i
    return s
""",
"""
def count_pos(lst):
    c = 0
    for x in lst:
        if x > 0:
            c += 1
    return c
""",
"""
def max_list(lst):
    m = lst[0]
    for x in lst:
        if x > m:
            m = x
    return m
""",
"""
def square_list(lst):
    return [x*x for x in lst]
""",
"""
def find_zero(lst):
    for x in lst:
        if x == 0:
            return True
    return False
""",
"""
def print_nums(n):
    for i in range(n):
        print(i)
""",
"""
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
""",
"""
def is_even(x):
    return x % 2 == 0
""",
"""
def sum_list(lst):
    return sum(lst)
""",
"""
def search(lst, x):
    for v in lst:
        if v == x:
            return True
    return False
""",
"""
def cube(x):
    return x*x*x
""",
"""
def all_positive(lst):
    for x in lst:
        if x <= 0:
            return False
    return True
""",
"""
def print_msg():
    print("Hello")
"""
,
"""
def power(a, b):
    return a ** b
""",
"""
def count_neg(lst):
    c = 0
    for x in lst:
        if x < 0:
            c += 1
    return c
""",
"""
def double(x):
    return x * 2
""",
"""
def sum_even(n):
    s = 0
    for i in range(n):
        if i % 2 == 0:
            s += i
    return s
""",
"""
def contains(lst, x):
    for v in lst:
        if v == x:
            return True
    return False
""",
"""
def is_positive(x):
    return x > 0
""",
"""
def square_sum(a, b):
    return a*a + b*b
""",
"""
def sum_list(lst):
    return sum(lst)
""",
"""
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
""",
"""
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
""",
"""
def read_file(name):
    f = open(name)
    return f.read()
""",
"""
n = 10
total = 0

for i in range(n):
    if i % 2 == 0:
        print(i)
        total += i

print("Sum:", total)
""",
"""
x = 5
if x > 0:
    print("Positive")
else:
    print("Negative")
""",
"""
a = 5
b = 7
c = a * a + b * b
print(c)
""",
"""
arr = [4, 1, 3, 2]
arr.sort()
print(arr)
""",
"""
arr = [2, 4, 6, 7, 8]
is_odd = False

for x in arr:
    if x % 2 != 0:
        is_odd = True
        break

print(is_odd)
""",
"""
arr = [3, -1, 5, 0, 7, -2]
count = 0

for x in arr:
    if x > 0:
        count += 1

print("Positive count:", count)
""",
"""
nums = [1, 2, 3, 4, 5, 6]
total = 0

for x in nums:
    total += x
    if total > 10:
        break

print("Total:", total)
""",
"""
values = [8, 3, 6, 2, 9]
min_val = values[0]

for x in values:
    if x < min_val:
        min_val = x

print("Min:", min_val)
""",
"""
nums = [1, 2, 3, 4]
product = 1

for x in nums:
    product *= x

print("Product:", product)
""",
"""
nums = [2, 4, 6, 8]
all_even = True

for x in nums:
    if x % 2 != 0:
        all_even = False
        break

print(all_even)
""",
"""
x = 7
if x > 5:
    print("Large")
else:
    print("Small")
""",
"""
arr = [4, 7, 1, 9]
found = False

for x in arr:
    if x == 7:
        found = True
        break

print(found)
""",
"""
x = 10
is_large = x > 100
print(is_large)
""",
"""
arr = [5, 6, 7]
count = 0

for x in arr:
    count += 1

print(count)
""",
"""
n = 20
total = 0

for i in range(n):
    if i % 2 != 0:
        total += i

print("Odd sum:", total)
""",
"""
text = "artificial intelligence"
count = 0

for ch in text:
    if ch in "aeiou":
        count += 1

print("Vowels:", count)
""",
"""
n = 17
is_prime = True

if n < 2:
    is_prime = False

for i in range(2, n):
    if n % i == 0:
        is_prime = False
        break

print(is_prime)
""",
"""
nums = [1, 3, 5, 7, 9]
total = 0

for x in nums:
    total += x
    if total > 10:
        break

print(total)
""",
"""
text = "machinelearning"
count = 0

for ch in text:
    count += 1

print(count)
""",
"""
n = 20
total = 0

for i in range(n):
    if i % 2 == 0:
        total += i

print("Sum:", total)
""",
"""
arr = [3, 7, 1, 9, 4]
max_val = arr[0]

for x in arr:
    if x > max_val:
        max_val = x

print(max_val)
""",
"""
arr = [2, 4, 6, 8, 10]
found = False

for x in arr:
    if x == 6:
        found = True
        break

print(found)
""",
"""
x = 5

if x > 0:
    print("Positive")
else:
    print("Negative")
""",
"""
nums = [1, 2, 3, 4, 5, 6]
total = 0

for x in nums:
    total += x
    if total > 10:
        break

print(total)
""",
"""
nums = [2, 4, 6, 8]
all_even = True

for x in nums:
    if x % 2 != 0:
        all_even = False
        break

print(all_even)
""",
"""
nums = [2, 4, 6, 8]
all_even = True

for x in nums:
    if x % 2 != 0:
        all_even = False
        break

print(all_even)
""",
"""
for i in range(5):
    print(i)
""",
"""
a = 4
b = 6
result = a * b
print(result)
""",
"""
arr = [4, 2, 1, 3]
arr.sort()
print(arr)
""",
"""
x = 10
flag = x > 5
print(flag)
""",
"""
n = 7
is_prime = True

for i in range(2, n):
    if n % i == 0:
        is_prime = False
        break

print(is_prime)
""",
"""
arr = [1, 3, 5, 7]
found = False

for x in arr:
    if x == 3:
        found = True
        break

print(found)
""",
"""
n = 20
total = 0
for i in range(n):
    if i % 2 == 0:
        total += i
print(total)
""",

"""
arr = [3, 8, 2, 9, 5]
max_val = arr[0]
for x in arr:
    if x > max_val:
        max_val = x
print(max_val)
""",

"""
arr = [2, 4, 6, 8]
found = False
for x in arr:
    if x == 6:
        found = True
        break
print(found)
""",

"""
nums = [1, 2, 3, 4, 5]
total = 0
for x in nums:
    total += x
    if total > 6:
        break
print(total)
""",

"""
arr = [-1, 3, 5, -2, 7]
count = 0
for x in arr:
    if x > 0:
        count += 1
print(count)
""",

"""
nums = [2, 4, 6, 8]
all_even = True
for x in nums:
    if x % 2 != 0:
        all_even = False
        break
print(all_even)
""",

"""
for i in range(5):
    print(i)
""",

"""
a = 4
b = 6
print(a * b)
""",

"""
arr = [4, 2, 1, 3]
arr.sort()
print(arr)
""",

"""
x = 10
flag = x > 5
print(flag)
""",
"""
n = 20
total = 0
for i in range(n):
    if i % 2 == 0:
        total += i
print(total)
""",
"""
arr = [3, 8, 2, 9, 5]
max_val = arr[0]
for x in arr:
    if x > max_val:
        max_val = x
print(max_val)
""",
"""
arr = [2, 4, 6, 8]
found = False
for x in arr:
    if x == 6:
        found = True
        break
print(found)
""",
"""
nums = [1, 2, 3, 4, 5]
total = 0
for x in nums:
    total += x
    if total > 6:
        break
print(total)
""",
"""
arr = [-1, 3, 5, -2, 7]
count = 0
for x in arr:
    if x > 0:
        count += 1
print(count)
""",
"""
nums = [2, 4, 6, 8]
all_even = True
for x in nums:
    if x % 2 != 0:
        all_even = False
        break
print(all_even)
""",
"""
for i in range(5):
    print(i)
""",
"""
a = 4
b = 6
print(a * b)
""",
"""
arr = [1, 2, 3, 4, 5]
count = 0
for x in arr:
    if x % 2 != 0:
        count += 1
print(count)
""",
"""
nums = [2, 5, 8, 1, 9]
total = 0
for x in nums:
    if x > 4:
        total += x
print(total)
""",
"""
arr = [3, 5, -2, 7]
has_negative = False
for x in arr:
    if x < 0:
        has_negative = True
        break
print(has_negative)
""",
"""
nums = [1, 2, 3, 4]
total = 0
for x in nums:
    total += x * x
print(total)
""",
"""
arr = [4, 6, 8, 9]
count = 0
for _ in arr:
    count += 1
print(count)
""",
"""
nums = [1, 2, 3, 4, 5]
for x in nums:
    print(x)
    if x == 3:
        break
""",
"""
arr = [5, 3, 9, 2, 8]
min_val = arr[0]
for x in arr:
    if x < min_val:
        min_val = x
print(min_val)
""",
"""
nums = [2, 4, 6]
valid = True
for x in nums:
    if x >= 10:
        valid = False
        break
print(valid)
""",
"""
for i in range(5):
    print(i)
""",
"""
x = 12
is_big = x > 10
print(is_big)
""",
"""
nums = [3, 6, 9, 12]
total = 0
for x in nums:
    total += x
print(total)
""",
"""
arr = [5, 1, 4, 2]
count = 0
for x in arr:
    if x > 3:
        count += 1
print(count)
""",
"""
nums = [1, 3, 5, 7]
flag = True
for x in nums:
    if x % 2 == 0:
        flag = False
        break
print(flag)
""",
"""
arr = [10, 20, 30]
total = 0
for x in arr:
    total += x * 2
print(total)
""",
"""
nums = [2, 4, 6, 8]
count = 0
for x in nums:
    count += 1
print(count)
""",
"""
arr = [7, 3, 9, 1]
max_val = arr[0]
for x in arr:
    if x > max_val:
        max_val = x
print(max_val)
""",
"""
nums = [1, 2, 3, 4, 5]
total = 0
for x in nums:
    if x < 4:
        total += x
print(total)
""",
"""
arr = [4, 6, 8, 10]
flag = False
for x in arr:
    if x == 6:
        flag = True
        break
print(flag)
""",
"""
for i in range(6):
    print(i * i)
""",
"""
x = 9
result = x + x
print(result)
""",
"""
nums = [5, 10, 15, 20]
total = 0
for x in nums:
    total += x
print(total)
""",
"""
arr = [1, -2, 3, -4, 5]
count = 0
for x in arr:
    if x < 0:
        count += 1
print(count)
""",
"""
nums = [2, 3, 4, 5]
product = 1
for x in nums:
    product *= x
print(product)
""",
"""
arr = [9, 4, 7, 2]
min_val = arr[0]
for x in arr:
    if x < min_val:
        min_val = x
print(min_val)
""",
"""
nums = [1, 2, 3, 4, 5]
flag = False
for x in nums:
    if x == 4:
        flag = True
        break
print(flag)
""",
"""
nums = [6, 8, 10]
all_even = True
for x in nums:
    if x % 2 != 0:
        all_even = False
        break
print(all_even)
""",
"""
arr = [3, 6, 9]
total = 0
for x in arr:
    total += x * x
print(total)
""",
"""
nums = [1, 3, 5, 7]
count = 0
for x in nums:
    count += 1
print(count)
""",
"""
for i in range(4):
    print(i + 1)
""",
"""
x = 14
result = x - 4
print(result)
""",
"""
nums = [1, 2, 3, 4, 5]
total = 0
for x in nums:
    total += x
print(total)
""",
"""
arr = [3, -2, 5, -7, 9]
count = 0
for x in arr:
    if x < 0:
        count += 1
print(count)
""",
"""
nums = [2, 4, 6, 8]
flag = True
for x in nums:
    if x % 2 != 0:
        flag = False
print(flag)
""",
"""
arr = [5, 3, 9, 1]
max_val = arr[0]
for x in arr:
    if x > max_val:
        max_val = x
print(max_val)
""",
"""
nums = [1, 2, 3, 4]
product = 1
for x in nums:
    product *= x
print(product)
""",
"""
n = 10
for i in range(n):
    if i % 2 == 0:
        print(i)
""",
"""
x = 7
if x > 0:
    print("Positive")
else:
    print("Negative")
""",
"""
nums = [1, 3, 5]
count = 0
for _ in nums:
    count += 1
print(count)
""",
"""
a = 5
b = 10
c = a + b
print(c)
""",
"""
nums = [2, 4, 6]
print(all(x % 2 == 0 for x in nums))
""",
"""
nums = [10, 20, 30]
s = 0
for i in range(len(nums)):
    s += nums[i]
print(s)
""",

"""
arr = [1, 4, 7, 9]
found = False
for x in arr:
    if x == 7:
        found = True
print(found)
""",

"""
nums = [5, 8, 12]
min_val = nums[0]
for x in nums:
    if x < min_val:
        min_val = x
print(min_val)
""",

"""
s = "hello"
count = 0
for ch in s:
    if ch in "aeiou":
        count += 1
print(count)
""",

"""
nums = [1, 2, 3, 4, 5]
even = []
for x in nums:
    if x % 2 == 0:
        even.append(x)
print(even)
""",

"""
n = 5
fact = 1
for i in range(1, n+1):
    fact *= i
print(fact)
""",

"""
nums = [3, 6, 9]
total = 0
for x in nums:
    total += x*x
print(total)
""",

"""
s = "madam"
rev = s[::-1]
print(s == rev)
""",

"""
nums = [1, 2, 3]
res = 0
for i in nums:
    res += i
print(res / len(nums))
""",

"""
a = 15
b = 20
if a > b:
    print(a)
else:
    print(b)
""",
"""
nums = [1, 2, 3, 4]
s = 0
for x in nums:
    s += x
print(s)
""",

"""
nums = [5, 10, 15]
p = 1
for x in nums:
    p *= x
print(p)
""",

"""
nums = [3, 6, 9]
cnt = 0
for x in nums:
    if x % 3 == 0:
        cnt += 1
print(cnt)
""",

"""
arr = [1, 4, 2, 8]
m = arr[0]
for x in arr:
    if x > m:
        m = x
print(m)
""",

"""
arr = [5, 3, 7]
m = arr[0]
for x in arr:
    if x < m:
        m = x
print(m)
""",

"""
s = "python"
count = 0
for c in s:
    count += 1
print(count)
""",

"""
nums = [1, 2, 3, 4, 5]
even = []
for x in nums:
    if x % 2 == 0:
        even.append(x)
print(even)
""",

"""
nums = [1, 2, 3, 4]
odd = []
for x in nums:
    if x % 2 != 0:
        odd.append(x)
print(odd)
""",

"""
nums = [2, 4, 6]
flag = True
for x in nums:
    if x % 2 != 0:
        flag = False
print(flag)
""",

"""
nums = [1, 3, 5]
flag = True
for x in nums:
    if x % 2 == 0:
        flag = False
print(flag)
""",

"""
n = 5
fact = 1
for i in range(1, n+1):
    fact *= i
print(fact)
""",

"""
nums = [2, 3, 4]
s = 0
for x in nums:
    s += x*x
print(s)
""",

"""
nums = [1, 2, 3]
avg = 0
for x in nums:
    avg += x
print(avg/len(nums))
""",

"""
s = "madam"
rev = ""
for c in s:
    rev = c + rev
print(s == rev)
""",

"""
a = 10
b = 20
if a > b:
    print(a)
else:
    print(b)
""",

"""
arr = [1, 2, 3]
found = False
for x in arr:
    if x == 2:
        found = True
print(found)
""",

"""
nums = [5, 6, 7]
total = 0
i = 0
while i < len(nums):
    total += nums[i]
    i += 1
print(total)
""",

"""
nums = [1, 2, 3]
i = 0
while i < len(nums):
    print(nums[i])
    i += 1
""",

"""
s = "hello"
vowels = 0
for c in s:
    if c in "aeiou":
        vowels += 1
print(vowels)
""",

"""
nums = [4, 5, 6]
res = []
for x in nums:
    res.append(x*2)
print(res)
""",

# -------- duplicated logic with variations --------

"""
nums = [10, 20]
print(nums[0] + nums[1])
""",

"""
nums = [2, 3]
print(nums[0] * nums[1])
""",

"""
x = 7
if x > 0:
    print("Positive")
else:
    print("Negative")
""",

"""
nums = [1, 2, 3]
print(len(nums))
""",

"""
nums = [3, 1, 2]
nums.sort()
print(nums)
""",

"""
nums = [1, 2, 3]
print(nums[::-1])
""",

"""
nums = [5, 6, 7]
print(nums[0])
""",

"""
nums = [5, 6, 7]
print(nums[-1])
""",

"""
nums = [1, 2, 3]
s = 0
for i in range(len(nums)):
    s += nums[i]
print(s)
""",

"""
nums = [2, 4, 6]
count = 0
for i in nums:
    count += 1
print(count)
""",

"""
nums = [1, 3, 5]
for i in range(len(nums)):
    print(nums[i])
""",

"""
nums = [1, 2, 3]
if len(nums) > 0:
    print("Not empty")
""",

"""
nums = [0, 1, 0]
zero = 0
for x in nums:
    if x == 0:
        zero += 1
print(zero)
""",

"""
s = "abc"
for i in range(len(s)):
    print(s[i])
""",

"""
nums = [2, 3, 4]
prod = 1
for x in nums:
    prod *= x
print(prod)
""",

"""
nums = [1, 2, 3]
print(sum(nums))
""",

"""
nums = [1, 2, 3]
print(min(nums))
""",

"""
nums = [1, 2, 3]
print(max(nums))
""",

"""
nums = [1, 2, 3]
print(all(x > 0 for x in nums))
""",

"""
nums = [0, 1, 2]
print(any(x == 0 for x in nums))
""",

"""
s = "level"
print(s == s[::-1])
""",

"""
nums = [1, 2, 3]
print(list(map(lambda x: x+1, nums)))
""",

"""
nums = [1, 2, 3]
print(list(filter(lambda x: x%2==0, nums)))
""",

"""
nums = [1, 2, 3]
print([x*x for x in nums])
""",

"""
nums = [1, 2, 3]
print([x for x in nums if x > 1])
""",

"""
nums = [1, 2, 3]
print(sum(x*x for x in nums))
""",
"""
nums = [1, 2, 3, 4]
s = 0
for x in nums:
    s += x
print(s)
""",

"""
nums = [1, 2, 3, 4]
p = 1
for x in nums:
    p *= x
print(p)
""",

"""
nums = [2, 4, 6]
flag = True
for x in nums:
    if x % 2 != 0:
        flag = False
print(flag)
""",

"""
nums = [1, 3, 5]
count = 0
for x in nums:
    if x % 2 == 0:
        count += 1
print(count)
""",

"""
s = "hello"
count = 0
for c in s:
    count += 1
print(count)
""",

"""
s = "hello"
vowels = 0
for c in s:
    if c in "aeiou":
        vowels += 1
print(vowels)
""",

"""
nums = [1, 2, 3]
print(min(nums))
""",

"""
nums = [1, 2, 3]
print(max(nums))
""",

"""
nums = [1, 2, 3]
print(sum(nums))
""",

"""
nums = [1, 2, 3]
print(len(nums))
""",

"""
a = 10
b = 5
print(a - b)
""",

"""
a = 10
b = 5
print(a / b)
""",

"""
n = 5
fact = 1
for i in range(1, n+1):
    fact *= i
print(fact)
""",

"""
n = 5
s = 0
for i in range(1, n+1):
    s += i
print(s)
""",

"""
nums = [1, 2, 3]
print(nums[::-1])
""",

"""
nums = [1, 2, 3]
nums.sort()
print(nums)
""",

"""
x = 7
if x > 0:
    print("Positive")
else:
    print("Negative")
""",

"""
x = 7
if x % 2 == 0:
    print("Even")
else:
    print("Odd")
""",

"""
nums = [1, 2, 3]
res = []
for x in nums:
    res.append(x * 2)
print(res)
""",

"""
nums = [1, 2, 3]
res = []
for x in nums:
    res.append(x * x)
print(res)"""
]

code_list_2 = [
    """
def f(x):
    for i in range(30):
        print(i)
""",

    """
def p(b):
    i = 0
    while i < 30:
        print(i)
        i += 1
""",

    """
def add(a, b):
        return a + b
""",
 """
def add(a, b):
        return a + b
""",
"""
def g(n):
    i = 0
    while i < n:
        print(i)
        i += 1
""",
"""def count(lst):
    i = 0
    while i < len(lst):
        print(lst[i])
        i += 1
""",
"""
def maximum(x, y):
    return x if x > y else y
""",
"""
def g(x):
    if x > 0:
        return x
""",
"""
def upper(s):
    return s.upper()
""",
"""
def print_nums(n):
    for i in range(n):
        print(i)
""",
"""
def show_even(n):
    i = 0
    while i < n:
        if i % 2 == 0:
            print(i)
        i += 1
""",
"""
def sum_n(n):
    s = 0
    i = 0
    while i < n:
        s = s + i
        i += 1
    return s
""",
"""
def count_pos(lst):
    c = 0
    i = 0
    while i < len(lst):
        if lst[i] > 0:
            c += 1
        i += 1
    return c
""",
"""
def max_list(lst):
    m = lst[0]
    i = 1
    while i < len(lst):
        if lst[i] > m:
            m = lst[i]
        i += 1
    return m
""",
"""
def square_list(lst):
    res = []
    for x in lst:
        res.append(x*x)
    return res
""",
"""
def find_zero(lst):
    i = 0
    while i < len(lst):
        if lst[i] == 0:
            return True
        i += 1
    return False
""",
"""
def square(x):
    return x * x
""",
"""
def print_even(n):
    for i in range(n):
        if i % 2 == 0:
            print(i)
""",
"""
def to_upper(s):
    return s.upper()
""",
"""
def length(lst):
    return len(lst)
""",
"""
def count(lst):
    return len(lst)
""",
"""
def negate(x):
    return -x
""",
"""
def any_positive(lst):
    for x in lst:
        if x > 0:
            return True
    return False
""",
"""
def return_msg():
    return "Hello"
""",
"""
def add(a, b):
    return a + b
""",
"""
def count_neg(lst):
    c = 0
    i = 0
    while i < len(lst):
        if lst[i] < 0:
            c += 1
        i += 1
    return c
""",
"""
def double(y):
    z = y + y
    return z
""",
"""
def sum_even(n):
    s = 0
    i = 0
    while i < n:
        if i % 2 == 0:
            s += i
        i += 1
    return s
""",
"""
def contains(lst, x):
    return x in lst
""",
"""
def is_positive(y):
    if y > 0:
        return True
    return False
""",
"""
def square_sum(x, y):
    s1 = x * x
    s2 = y * y
    return s1 + s2
""",
"""
def print_list(lst):
    for x in lst:
        print(x)
""",
"""
def reverse(s):
    return s[::-1]
""",
"""
def max_val(a, b):
    return a if a > b else b
""",
"""
def square(x):
    return x * x
""",
"""
n = 10
total = 0
i = 0

while i < n:
    if i % 2 == 0:
        print(i)
        total = total + i
    i += 1

print("Sum:", total)
""",
"""
n = 10
for i in range(n):
    print(i)
""",
"""
s = "hello world"
result = s.upper()
print(result)
""",
"""
arr = [4, 1, 3, 2]
count = 0

for x in arr:
    count += 1

print(count)
""",
"""
x = 7
is_odd = x % 2 != 0
print(is_odd)
""",
"""
arr = [3, -1, 5, 0, 7, -2]
count = 0
i = 0

while i < len(arr):
    if arr[i] > 0:
        count = count + 1
    i += 1

print("Positive count:", count)
""",
"""
nums = [1, 2, 3, 4, 5, 6]
total = 0
i = 0

while i < len(nums):
    total = total + nums[i]
    if total > 10:
        break
    i += 1

print("Total:", total)
""",
"""
values = [8, 3, 6, 2, 9]
min_val = values[0]
i = 1

while i < len(values):
    if values[i] < min_val:
        min_val = values[i]
    i += 1

print("Min:", min_val)
""",
"""
nums = [1, 2, 3, 4]
product = 1
i = 0

while i < len(nums):
    product = product * nums[i]
    i += 1

print("Product:", product)
""",
"""
nums = [2, 4, 6, 8]
all_even = True
i = 0

while i < len(nums):
    if nums[i] % 2 != 0:
        all_even = False
        break
    i += 1

print(all_even)
""",
"""
nums = [1, 2, 3, 4]
total = 0

for x in nums:
    total += x

print(total)
""",
"""
arr = [4, 7, 1, 9]
arr.sort()
print(arr)
""",
"""
r = 3
area = 3.14 * r * r
print(area)
""",
"""
text = "hello"
upper = text.upper()
print(upper)
""",
"""
n = 20
total = 0
i = 0

while i < n:
    if i % 2 != 0:
        total = total + i
    i += 1

print("Odd sum:", total)
""",
"""
text = "artificial intelligence"
result = text.upper()
print(result)
""",
"""
x = 16
print(x % 2 == 0)
""",
"""
nums = [1, 3, 5, 7, 9]
total = 0

for x in nums:
    total += x

print(total)
""",
"""
text = "machinelearning"
print(len(text))
""",
"""
n = 20
total = 0

for i in range(n):
    if i % 2 == 0:
        total += i

print("Sum:", total)
""",
"""
arr = [3, 7, 1, 9, 4]
max_val = arr[0]

for x in arr:
    if x > max_val:
        max_val = x

print(max_val)
""",
"""
arr = [2, 4, 6, 8, 10]
found = False

for x in arr:
    if x == 6:
        found = True
        break

print(found)
""",
"""
x = 5

if x > 0:
    print("Positive")
else:
    print("Negative")
""",
"""
nums = [1, 2, 3, 4, 5, 6]
total = 0

for x in nums:
    total += x
    if total > 10:
        break

print(total)
""",
"""
nums = [2, 4, 6, 8]
all_even = True

for x in nums:
    if x % 2 != 0:
        all_even = False
        break

print(all_even)
""",
"""
n = 20
total = 0
i = 0

while i < n:
    if i % 2 == 0:
        total = total + i
    i += 1

print(total)
""",
"""
x = 5
if x > 0:
    print("Positive")
else:
    print("Negative")
""",
"""
text = "hello"
print(text.upper())
""",
"""
arr = [4, 2, 1, 3]
count = 0

for x in arr:
    count += 1

print(count)
""",
"""
for i in range(3):
    print(i)
""",
"""
nums = [1, 2, 3, 4]
total = 0

for x in nums:
    total += x

print(total)
""",
"""
arr = [1, 3, 5, 7]
arr.reverse()
print(arr)
""",

"""
n = 20
total = 0
i = 0
while i < n:
    if i % 2 == 0:
        total = total + i
    i += 1
print(total)
""",

"""
arr = [3, 8, 2, 9, 5]
i = 1
max_val = arr[0]
while i < len(arr):
    if arr[i] > max_val:
        max_val = arr[i]
    i += 1
print(max_val)
""",

"""
arr = [2, 4, 6, 8]
i = 0
found = False
while i < len(arr):
    if arr[i] == 6:
        found = True
        break
    i += 1
print(found)
""",

"""
nums = [1, 2, 3, 4, 5]
i = 0
total = 0
while i < len(nums):
    total = total + nums[i]
    if total > 6:
        break
    i += 1
print(total)
""",

"""
arr = [-1, 3, 5, -2, 7]
count = 0
i = 0
while i < len(arr):
    if arr[i] > 0:
        count += 1
    i += 1
print(count)
""",

"""
nums = [2, 4, 6, 8]
i = 0
all_even = True
while i < len(nums):
    if nums[i] % 2 != 0:
        all_even = False
        break
    i += 1
print(all_even)
""",

"""
x = 5
if x > 0:
    print("Positive")
else:
    print("Negative")
""",

"""
text = "hello"
print(text.upper())
""",

"""
arr = [4, 2, 1, 3]
count = 0
for x in arr:
    count += 1
print(count)
""",

"""
for i in range(3):
    print(i)
""",
"""
n = 20
total = 0
i = 0
while i < n:
    if i % 2 == 0:
        total = total + i
    i += 1
print(total)
""",
"""
arr = [3, 8, 2, 9, 5]
i = 1
max_val = arr[0]
while i < len(arr):
    if arr[i] > max_val:
        max_val = arr[i]
    i += 1
print(max_val)
""",
"""
arr = [2, 4, 6, 8]
i = 0
found = False
while i < len(arr):
    if arr[i] == 6:
        found = True
        break
    i += 1
print(found)
""",
"""
nums = [1, 2, 3, 4, 5]
i = 0
total = 0
while i < len(nums):
    total = total + nums[i]
    if total > 6:
        break
    i += 1
print(total)
""",
"""
arr = [-1, 3, 5, -2, 7]
count = 0
i = 0
while i < len(arr):
    if arr[i] > 0:
        count += 1
    i += 1
print(count)
""",
"""
nums = [2, 4, 6, 8]
i = 0
all_even = True
while i < len(nums):
    if nums[i] % 2 != 0:
        all_even = False
        break
    i += 1
print(all_even)
""",
"""
x = 5
if x > 0:
    print("Positive")
else:
    print("Negative")
""",
"""
text = "hello"
print(text.upper())
""",
"""
arr = [1, 2, 3, 4, 5]
count = 0
i = 0
while i < len(arr):
    if arr[i] % 2 != 0:
        count = count + 1
    i += 1
print(count)
""",
"""
nums = [2, 5, 8, 1, 9]
total = 0
i = 0
while i < len(nums):
    if nums[i] > 4:
        total = total + nums[i]
    i += 1
print(total)
""",
"""
arr = [3, 5, -2, 7]
i = 0
has_negative = False
while i < len(arr):
    if arr[i] < 0:
        has_negative = True
        break
    i += 1
print(has_negative)
""",
"""
nums = [1, 2, 3, 4]
total = 0
i = 0
while i < len(nums):
    total = total + nums[i] * nums[i]
    i += 1
print(total)
""",
"""
arr = [4, 6, 8, 9]
i = 0
count = 0
while i < len(arr):
    count += 1
    i += 1
print(count)
""",
"""
nums = [1, 2, 3, 4, 5]
i = 0
while i < len(nums):
    print(nums[i])
    if nums[i] == 3:
        break
    i += 1
""",
"""
arr = [5, 3, 9, 2, 8]
i = 1
min_val = arr[0]
while i < len(arr):
    if arr[i] < min_val:
        min_val = arr[i]
    i += 1
print(min_val)
""",
"""
nums = [2, 4, 6]
i = 0
valid = True
while i < len(nums):
    if nums[i] >= 10:
        valid = False
        break
    i += 1
print(valid)
""",
"""
a = 5
b = 7
print(a * b)
""",
"""
i = 0
while i < 4:
    print(i)
    i += 1
""",
"""
nums = [3, 6, 9, 12]
i = 0
total = 0
while i < len(nums):
    total = total + nums[i]
    i += 1
print(total)
""",
"""
arr = [5, 1, 4, 2]
i = 0
count = 0
while i < len(arr):
    if arr[i] > 3:
        count += 1
    i += 1
print(count)
""",
"""
nums = [1, 3, 5, 7]
i = 0
flag = True
while i < len(nums):
    if nums[i] % 2 == 0:
        flag = False
        break
    i += 1
print(flag)
""",
"""
arr = [10, 20, 30]
i = 0
total = 0
while i < len(arr):
    total = total + arr[i] * 2
    i += 1
print(total)
""",
"""
nums = [2, 4, 6, 8]
i = 0
count = 0
while i < len(nums):
    count += 1
    i += 1
print(count)
""",
"""
arr = [7, 3, 9, 1]
i = 1
max_val = arr[0]
while i < len(arr):
    if arr[i] > max_val:
        max_val = arr[i]
    i += 1
print(max_val)
""",
"""
nums = [1, 2, 3, 4, 5]
i = 0
total = 0
while i < len(nums):
    if nums[i] < 4:
        total = total + nums[i]
    i += 1
print(total)
""",
"""
arr = [4, 6, 8, 10]
i = 0
flag = False
while i < len(arr):
    if arr[i] == 6:
        flag = True
        break
    i += 1
print(flag)
""",
"""
i = 0
while i < 6:
    print(i * i)
    i += 1
""",
"""
y = 9
result = y * 2
print(result)
""",
"""
nums = [5, 10, 15, 20]
i = 0
total = 0
while i < len(nums):
    total = total + nums[i]
    i += 1
print(total)
""",
"""
arr = [1, -2, 3, -4, 5]
i = 0
count = 0
while i < len(arr):
    if arr[i] < 0:
        count += 1
    i += 1
print(count)
""",
"""
nums = [2, 3, 4, 5]
i = 0
product = 1
while i < len(nums):
    product = product * nums[i]
    i += 1
print(product)
""",
"""
arr = [9, 4, 7, 2]
i = 1
min_val = arr[0]
while i < len(arr):
    if arr[i] < min_val:
        min_val = arr[i]
    i += 1
print(min_val)
""",
"""
nums = [1, 2, 3, 4, 5]
i = 0
flag = False
while i < len(nums):
    if nums[i] == 4:
        flag = True
        break
    i += 1
print(flag)
""",
"""
nums = [6, 8, 10]
i = 0
all_even = True
while i < len(nums):
    if nums[i] % 2 != 0:
        all_even = False
        break
    i += 1
print(all_even)
""",
"""
arr = [3, 6, 9]
i = 0
total = 0
while i < len(arr):
    total = total + arr[i] * arr[i]
    i += 1
print(total)
""",
"""
nums = [1, 3, 5, 7]
i = 0
count = 0
while i < len(nums):
    count += 1
    i += 1
print(count)
""",
"""
i = 0
while i < 4:
    print(i + 1)
    i += 1
""",
"""
y = 14
result = y / 2
print(result)
""",
"""
nums = [1, 2, 3, 4, 5]
i = 0
total = 0
while i < len(nums):
    total = total + nums[i]
    i += 1
print(total)
""",
"""
arr = [3, -2, 5, -7, 9]
i = 0
count = 0
while i < len(arr):
    if arr[i] < 0:
        count += 1
    i += 1
print(count)
""",
"""
nums = [2, 4, 6, 8]
i = 0
flag = True
while i < len(nums):
    if nums[i] % 2 != 0:
        flag = False
    i += 1
print(flag)
""",
"""
arr = [5, 3, 9, 1]
i = 1
max_val = arr[0]
while i < len(arr):
    if arr[i] > max_val:
        max_val = arr[i]
    i += 1
print(max_val)
""",
"""
nums = [1, 2, 3, 4]
i = 0
product = 1
while i < len(nums):
    product = product * nums[i]
    i += 1
print(product)
""",
"""
n = 10
i = 0
while i < n:
    if i % 2 == 0:
        print(i)
    i += 1
""",
"""
x = -7
if x > 0:
    print("Positive")
else:
    print("Negative")
""",
"""
nums = [1, 3, 5]
i = 0
count = 0
while i < len(nums):
    count += 1
    i += 1
print(count)
""",
"""
a = 5
b = 10
print(a * b)
""",
"""
nums = [1, 3, 5]
print(sum(nums))
""",
"""
nums = [10, 20, 30]
print(sum(nums))
""",

"""
arr = [1, 4, 7, 9]
print(7 in arr)
""",

"""
nums = [5, 8, 12]
print(min(nums))
""",

"""
s = "hello"
print(sum(1 for c in s if c in "aeiou"))
""",

"""
nums = [1, 2, 3, 4, 5]
print(list(filter(lambda x: x % 2 == 0, nums)))
""",

"""
import math
print(math.factorial(5))
""",

"""
nums = [3, 6, 9]
print(sum(map(lambda x: x*x, nums)))
""",

"""
s = "madam"
i = 0
j = len(s) - 1
flag = True
while i < j:
    if s[i] != s[j]:
        flag = False
        break
    i += 1
    j -= 1
print(flag)
""",

"""
nums = [1, 2, 3]
avg = sum(nums) / len(nums)
print(avg)
""",

"""
a = 15
b = 20
print(max(a, b))
""",
"""
nums = [1, 2, 3, 4]
print(sum(nums))
""",

"""
from functools import reduce
nums = [5, 10, 15]
print(reduce(lambda a,b:a*b, nums))
""",

"""
nums = [3, 6, 9]
print(len(nums))
""",

"""
arr = [1, 4, 2, 8]
print(max(arr))
""",

"""
arr = [5, 3, 7]
print(min(arr))
""",

"""
s = "python"
print(len(s))
""",

"""
nums = [1, 2, 3, 4, 5]
print([x for x in nums if x % 2 == 0])
""",

"""
nums = [1, 2, 3, 4]
print([x for x in nums if x % 2 != 0])
""",

"""
nums = [2, 4, 6]
print(all(x % 2 == 0 for x in nums))
""",

"""
nums = [1, 3, 5]
print(all(x % 2 != 0 for x in nums))
""",

"""
import math
print(math.factorial(5))
""",

"""
nums = [2, 3, 4]
print(sum(x*x for x in nums))
""",

"""
nums = [1, 2, 3]
print(sum(nums)/len(nums))
""",

"""
s = "madam"
i,j = 0,len(s)-1
ok = True
while i < j:
    if s[i] != s[j]:
        ok = False
        break
    i+=1; j-=1
print(ok)
""",

"""
a = 10
b = 20
print(max(a,b))
""",

"""
arr = [1, 2, 3]
print(2 in arr)
""",

"""
nums = [5, 6, 7]
print(sum(nums))
""",

"""
nums = [1, 2, 3]
for x in nums:
    print(x)
""",

"""
s = "hello"
print(sum(1 for c in s if c in "aeiou"))
""",

"""
nums = [4, 5, 6]
print(list(map(lambda x: x*2, nums)))
""",

"""
nums = [10, 20]
print(sum(nums))
""",

"""
nums = [2, 3]
print(nums[0] * nums[1])
""",

"""
x = 7
print("Positive" if x > 0 else "Negative")
""",

"""
nums = [1, 2, 3]
print(len(nums))
""",

"""
nums = [3, 1, 2]
print(sorted(nums))
""",

"""
nums = [1, 2, 3]
print(list(reversed(nums)))
""",

"""
nums = [5, 6, 7]
print(min(nums))
""",

"""
nums = [5, 6, 7]
print(max(nums))
""",

"""
nums = [1, 2, 3]
print(sum(nums))
""",

"""
nums = [2, 4, 6]
print(len(nums))
""",

"""
nums = [1, 3, 5]
print(*nums, sep="\\n")
""",

"""
nums = [1, 2, 3]
print(bool(nums))
""",

"""
nums = [0, 1, 0]
print(nums.count(0))
""",

"""
s = "abc"
print(*s, sep="\\n")
""",

"""
nums = [2, 3, 4]
print(reduce(lambda a,b:a*b, nums))
""",

"""
nums = [1, 2, 3]
print(sum(nums))
""",

"""
nums = [1, 2, 3]
print(min(nums))
""",

"""
nums = [1, 2, 3]
print(max(nums))
""",

"""
nums = [1, 2, 3]
print(all(x > 0 for x in nums))
""",

"""
nums = [0, 1, 2]
print(0 in nums)
""",

"""
s = "level"
print(s == s[::-1])
""",

"""
nums = [1, 2, 3]
print([x+1 for x in nums])
""",

"""
nums = [1, 2, 3]
print([x for x in nums if x%2==0])
""",

"""
nums = [1, 2, 3]
print([x*x for x in nums])
""",

"""
nums = [1, 2, 3]
print([x for x in nums if x > 1])
""",

"""
nums = [1, 2, 3]
print(sum(map(lambda x:x*x, nums)))
""",
"""
nums = [1, 2, 3, 4]
p = 1
for x in nums:
    p *= x
print(p)
""",

"""
nums = [1, 2, 3, 4]
s = 0
for x in nums:
    s += x
print(s)
""",

"""
nums = [2, 4, 6]
print(any(x % 2 != 0 for x in nums))
""",

"""
nums = [1, 3, 5]
print(len(nums))
""",

"""
s = "hello"
print(s[::-1])
""",

"""
s = "hello"
print(len(s))
""",

"""
nums = [1, 2, 3]
print(max(nums))
""",

"""
nums = [1, 2, 3]
print(min(nums))
""",

"""
nums = [1, 2, 3]
print(len(nums))
""",

"""
nums = [1, 2, 3]
print(sum(nums))
""",

"""
a = 10
b = 5
print(a + b)
""",

"""
a = 10
b = 5
print(a * b)
""",

"""
import math
print(math.factorial(4))
""",

"""
n = 5
print(n * (n + 1) // 2)
""",

"""
nums = [1, 2, 3]
print(nums)
""",

"""
nums = [1, 2, 3]
print(sorted(nums, reverse=True))
""",

"""
x = 7
print("Odd")
""",

"""
x = 7
print("Positive")
""",

"""
nums = [1, 2, 3]
print([x + 2 for x in nums])
""",

"""
nums = [1, 2, 3]
print([x - 1 for x in nums])
"""


]

labels = [
    1,  # similar (while vs for)
    1,  # similar (for vs while)
    0,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1, 
    1,  
    1, 
    1,  
    1,  
    1, 
    0,  
    0,  
    0,  
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
     1,  
    1,  
    1, 
    1,  
    1,  
    1, 
    1,  
    1,  
    1,  
    1,
    1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1,1,1,1,1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # if-else max vs max()


   # boolean vs loop
    
print(len(code_list_1), len(code_list_2), len(labels))
assert len(code_list_1) == len(code_list_2) == len(labels)


# =====================================================
# 2️⃣ CREATE DATASET
# =====================================================

def create_dataset(code1_list, code2_list, labels, filename="dataset/dataset.csv"):
    rows = []

    for c1, c2, lbl in zip(code1_list, code2_list, labels):
        features = extract_features(c1, c2)
        features["Label"] = lbl
        rows.append(features)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("✅ Dataset created successfully")
    print("📁 File:", filename)
    print("📊 Total samples:", len(rows))
def create_feature_dataset(code1_list, code2_list, labels, filename="dataset/features.csv"):
    rows = []

    for c1, c2, lbl in zip(code1_list, code2_list, labels):
        features = extract_features(c1, c2)
        features["Label"] = lbl
        rows.append(features)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("✅ Feature dataset created:", filename)
def create_raw_dataset(code1_list, code2_list, labels, filename="dataset/raw_data.csv"):
    rows = []

    for c1, c2, lbl in zip(code1_list, code2_list, labels):
        rows.append({
            "Code1_Raw": c1.strip(),
            "Code2_Raw": c2.strip(),
            "Label": lbl
        })

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Code1_Raw", "Code2_Raw", "Label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("✅ Raw dataset created:", filename)


# =====================================================
# 3️⃣ RUN
# =====================================================

create_dataset(code_list_1, code_list_2, labels)
create_feature_dataset(code_list_1, code_list_2, labels)
create_raw_dataset(code_list_1, code_list_2, labels)
