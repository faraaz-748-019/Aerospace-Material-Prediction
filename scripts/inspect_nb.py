import json

with open('notebooks/02_mechanical_analysis.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    print(f"=== Cell {i} ({ctype}) ===")
    source = ''.join(cell['source'])
    print(source[:250])
    if 'outputs' in cell and cell['outputs']:
        print(f"  [Outputs: {len(cell['outputs'])}]")
        for out in cell['outputs']:
            if out.get('output_type') == 'stream':
                print("  STREAM:", ''.join(out.get('text', []))[:200].replace('\n', ' '))
            elif out.get('output_type') == 'execute_result':
                print("  EXEC_RES:", str(out.get('data', {}).get('text/plain', ''))[:200].replace('\n', ' '))
