import bpy
import os

output_dir = r"C:\\Users\Miles\PycharmProjects\F1\blender_scripts"
os.makedirs(output_dir, exist_ok=True)

extracted = []

for text in bpy.data.texts:
    content = text.as_string()
    safe_name = "".join(c if c.isalnum() or c in " ._-" else "_" for c in text.name)
    if not safe_name.endswith(".py"):
        safe_name += ".py"
    out_path = os.path.join(output_dir, safe_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    extracted.append((text.name, len(content.splitlines()), out_path))

print(f"\n=== Extracted {len(extracted)} script(s) ===")
for name, lines, path in extracted:
    print(f"  {name}  ({lines} lines)  →  {path}")

if not extracted:
    print("No text blocks found in this .blend file.")
