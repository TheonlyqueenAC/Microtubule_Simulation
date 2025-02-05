# Import necessary classes from pylatex
from pylatex import Document
from pylatex.utils import NoEscape

# Define your LaTeX content as a string
latex_code = r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}

\section{Equations}
Here is an equation: 
\[
E = mc^2
\]
and another one:
\begin{align}
  a &= b + c \\
  d &= e + f
\end{align}

\end{document}
"""

# Create a LaTeX document object
doc = Document()

# Append your LaTeX code to the document
doc.append(NoEscape(latex_code))

# Generate PDF (will save it as 'output.pdf' in the current directory)
doc.generate_pdf('output', clean_tex=False)