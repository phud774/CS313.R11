# .latexmkrc — Build configuration for XeLaTeX
$pdf_mode = 5;          # 5 = xelatex
$xelatex = 'xelatex -interaction=nonstopmode -halt-on-error %O %S';
$biber = 'biber %O %S';
$clean_ext = 'nav snm vrb';
