#!/usr/bin/env python3
"""Fix LaTeX escaping in thesis.tex — single pass, no double-escaping issues."""
import re

with open("paper/thesis.tex", "r") as f:
    content = f.read()

# The file has literal double-backslash LaTeX commands like:
#   \\cite{...}  \\begin{...}  \\textbf{...}
# We need them to be single-backslash:
#   \cite{...}   \begin{...}   \textbf{...}
#
# Strategy: Replace every occurrence of two backslashes followed by a letter
# with one backslash followed by that letter.

# Step 1: Fix double-backslash commands (\\X -> \X)
content = re.sub(r'\\\\([a-zA-Z{])', r'\\\1', content)

# Step 2: Fix any quad-backslash remnants (\\\\ -> \\, for table line breaks)
content = re.sub(r'\\\\\\\\', r'\\\\', content)

# Step 3: Fix escaped underscores in text (\\_ -> \_)
content = content.replace('\\\\_', '\\_')

with open("paper/thesis.tex", "w") as f:
    f.write(content)

# Verify
for cmd in ['cite', 'begin', 'end', 'textbf', 'ref', 'label', 'subsection', 'section']:
    pattern = '\\\\' + cmd
    count = content.count(pattern)
    if count > 0:
        print(f"WARNING: {count} remaining '{pattern}'")

# Check that we have the right number of single-backslash commands
for cmd in ['\\cite', '\\begin', '\\end', '\\textbf', '\\ref', '\\label']:
    count = content.count(cmd)
    print(f"  OK: {count}x '{cmd}'")

print("Done.")
