.PHONY: graphs pdf clean

# Generate PDF from Graphviz dot files
graphs: model-diagram.pdf timestep-flowchart.pdf

model-diagram.pdf: model-diagram.dot
	dot -Tpdf model-diagram.dot -o model-diagram.pdf

timestep-flowchart.pdf: timestep-flowchart.dot
	dot -Tpdf timestep-flowchart.dot -o timestep-flowchart.pdf

# Compile LaTeX to PDF (requires the graphs to be built first)
pdf: graphs
	pdflatex -interaction=nonstopmode proposal.tex

clean:
	rm -f model-diagram.pdf timestep-flowchart.pdf proposal.pdf proposal.aux proposal.log proposal.fdb_latexmk proposal.fls
