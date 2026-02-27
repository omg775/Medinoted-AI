with open("app.py", "r") as f:
    lines = f.readlines()

for i in range(1207, 1308):
    if lines[i].strip():
        lines[i] = "    " + lines[i]

for i in range(1316, 1421):
    if lines[i].strip():
        lines[i] = "    " + lines[i]

with open("app.py", "w") as f:
    f.writelines(lines)
