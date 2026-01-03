from flask import Flask, render_template, request, jsonify
import ast
import networkx as nx
from collections import Counter
from math import sqrt
import tokenize, io, keyword
import pickle
import pandas as pd

app = Flask(__name__)


with open("model/plagiarism_model.pkl", "rb") as f:
    model = pickle.load(f)

# =====================================================
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
       
        pass


    total_tokens_safe = total_tokens if total_tokens > 0 else 1

    keyword_ratio = len(unique_keywords) / total_tokens_safe
    operator_ratio = len(unique_operators) / total_tokens_safe

    return {
        "Keyword_Ratio": keyword_ratio,
        "Operator_Ratio": operator_ratio,
        "Unique_Keyword_Count": len(unique_keywords),
        "Unique_Operator_Count": len(unique_operators)
    }



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

    
    features["A_LOC"] = loc1
    features["B_LOC"] = loc2

   
    for k in token_feats1:
        features[f"A_{k}"] = token_feats1[k]
        features[f"B_{k}"] = token_feats2[k]


  
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/compare",methods=["POST"])
def compare():
    data=request.get_json()
    code1=data.get("code1","")
    code2=data.get("code2","")

    feats=extract_features(code1,code2)
    X=pd.DataFrame([feats])
    print(X)
    pred=model.predict(X)[0]
    prob=model.predict_proba(X)[0][1]
    print(pred,prob)
    return jsonify({
    "prediction": "SIMILAR" if pred == 1 else "NOT SIMILAR",
    "confidence": round(float(prob), 4),
    "percentage": round(float(prob) * 100, 2)
})



if __name__=="__main__":
    app.run(debug=True)
