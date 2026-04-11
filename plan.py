import ast

def check_syntax(file_path):
    with open(file_path, "r") as f:
        src = f.read()
    try:
        ast.parse(src)
        print(f"{file_path} Syntax OK")
    except SyntaxError as e:
        print(f"Syntax Error in {file_path}: {e}")

check_syntax("src/efv_nn/ppc_core.py")
check_syntax("src/efv_nn/ppc_gnn.py")
