import sys, base64, json, pathlib
nb = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
for f in out.glob("*.png"): f.unlink()
k = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    for o in c.get("outputs", []):
        if o.get("output_type") == "error": print("ERROR in cell", i, o["ename"], o["evalue"])
        d = o.get("data", {})
        if "image/png" in d:
            b = base64.b64decode(d["image/png"]); k += 1
            (out / f"c{i:02d}_{k}.png").write_bytes(b); print(f"cell {i}: png {len(b)//1024} KB")
        for t in ("text/plain",):
            if o.get("output_type") == "stream": pass
