import ast
import networkx as nx
import numpy as np
from collections import Counter
from math import sqrt

# =====================================================
# COSINE SIMILARITY
# =====================================================
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1, float), np.array(v2, float)
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / d) if d else 0.0


# =====================================================
# AST FEATURES (25-FEATURE DESIGN)
# =====================================================
class ASTAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.c = Counter()
        self.total = 0
        self.depth = 0
        self.max_depth = 0

    def generic_visit(self, node):
        self.total += 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        super().generic_visit(node)
        self.depth -= 1

    def visit_If(self,n): self.c["If"]+=1; self.generic_visit(n)
    def visit_For(self,n): self.c["Loop"]+=1; self.generic_visit(n)
    def visit_While(self,n): self.c["Loop"]+=1; self.generic_visit(n)
    def visit_Assign(self,n): self.c["Assign"]+=1; self.generic_visit(n)
    def visit_AugAssign(self,n): self.c["Assign"]+=1; self.generic_visit(n)
    def visit_Return(self,n): self.c["Return"]+=1; self.generic_visit(n)
    def visit_Compare(self,n): self.c["Compare"]+=1; self.generic_visit(n)

def ast_vector(code):
    t = ast.parse(code)
    a = ASTAnalyzer()
    a.visit(t)
    tot = a.total or 1

    return [
        a.c["If"]/tot,
        a.c["Loop"]/tot,
        a.c["Assign"]/tot,
        a.c["Return"]/tot,
        a.c["Compare"]/tot,
        a.max_depth,
        len(a.c)/tot
    ]


# =====================================================
# CFG FEATURES
# =====================================================
class CFG(ast.NodeVisitor):
    def __init__(self):
        self.g = nx.DiGraph()
        self.i = 0
        self.cur = self.new()
        self.exit = self.new()

    def new(self):
        n = self.i
        self.g.add_node(n)
        self.i += 1
        return n

    def e(self,u,v): self.g.add_edge(u,v)

    def simple(self):
        n = self.new()
        self.e(self.cur,n)
        self.cur = n

    def visit_Assign(self,n): self.simple()
    def visit_AugAssign(self,n): self.simple()
    def visit_Expr(self,n): self.simple()

    def visit_For(self,n): self.visit_While(n)

    def visit_While(self,n):
        c = self.new()
        self.e(self.cur,c)
        self.cur = c
        for s in n.body: self.visit(s)
        self.e(self.cur,c)
        a = self.new()
        self.e(c,a)
        self.cur = a

    def visit_If(self,n):
        c = self.new()
        self.e(self.cur,c)
        a = self.new()
        self.cur = c
        for s in n.body: self.visit(s)
        self.e(self.cur,a)
        self.e(c,a)
        self.cur = a

    def visit_Return(self,n):
        r = self.new()
        self.e(self.cur,r)
        self.e(r,self.exit)
        self.cur = self.exit


def cfg_vector(code):
    t = ast.parse(code)
    b = CFG()
    b.visit(t)
    if b.cur != b.exit:
        b.e(b.cur,b.exit)

    N = b.g.number_of_nodes()
    E = b.g.number_of_edges()
    cyclo = E - N + 2
    branch_density = cyclo / N if N else 0

    return [N, E, cyclo, branch_density]


# =====================================================
# FLOW PATH SIMILARITY
# =====================================================
class FlowCFG(ast.NodeVisitor):
    def __init__(self):
        self.g = nx.DiGraph()
        self.lbl = {}
        self.i = 0
        self.entry = self.new("ENTRY")
        self.exit = self.new("EXIT")
        self.cur = self.entry

    def new(self,l):
        n = self.i
        self.g.add_node(n)
        self.lbl[n] = l
        self.i += 1
        return n

    def e(self,u,v): self.g.add_edge(u,v)

    def visit_Assign(self,n): self._stmt()
    def visit_AugAssign(self,n): self._stmt()
    def visit_Expr(self,n): self._stmt()

    def _stmt(self):
        if self.lbl[self.cur] == "ENTRY": return
        n = self.new("Stmt")
        self.e(self.cur,n)
        self.cur = n

    def visit_For(self,n): self._loop(n)
    def visit_While(self,n): self._loop(n)

    def _loop(self,n):
        c = self.new("Loop")
        self.e(self.cur,c)
        self.cur = c
        for s in n.body: self.visit(s)
        self.e(self.cur,c)
        a = self.new("AfterLoop")
        self.e(c,a)
        self.cur = a

    def visit_If(self,n):
        c = self.new("If")
        self.e(self.cur,c)
        a = self.new("AfterIf")
        self.cur = c
        for s in n.body: self.visit(s)
        self.e(self.cur,a)
        self.e(c,a)
        self.cur = a

    def visit_Return(self,n):
        r = self.new("Return")
        self.e(self.cur,r)
        self.e(r,self.exit)
        self.cur = self.exit


def flow_similarity(code1, code2):
    def paths(code):
        t = ast.parse(code)
        b = FlowCFG()
        b.visit(t)
        if b.cur != b.exit:
            b.e(b.cur,b.exit)
        res = set()
        def dfs(n,p):
            if len(p)==3 or b.g.out_degree(n)==0:
                res.add(tuple(p)); return
            for x in b.g.successors(n):
                dfs(x,p+[b.lbl[x]])
        dfs(b.entry,["ENTRY"])
        return res

    return len(paths(code1) & paths(code2)) / max(len(paths(code1) | paths(code2)), 1)


# =====================================================
# FINAL COSINE SIMILARITY (25 FEATURES)
# =====================================================
def cosine_code_similarity(code1, code2):
    ast1 = ast_vector(code1)
    ast2 = ast_vector(code2)

    cfg1 = cfg_vector(code1)
    cfg2 = cfg_vector(code2)

    flow = flow_similarity(code1, code2)

    # build 25-d vectors
    vec1 = ast1 + cfg1 + [flow]
    vec2 = ast2 + cfg2 + [flow]

    return cosine_similarity(vec1, vec2)


# =====================================================
# TEST
# =====================================================
if __name__ == "__main__":

    code1 = """
def f(n):
    for i in range(n):
        print(i)
"""

    code2 = """
def g(n):
    i = 0
    while i < n:
        print(i)
        i += 1
"""

    code3 = """
def h(x):
    print("Hello world")
"""

    print("for vs while:", cosine_code_similarity(code1, code2))
    print("loop vs if :", cosine_code_similarity(code1, code3))
