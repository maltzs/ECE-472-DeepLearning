fit: spiral.py
	black spiral.py
	flake8 --ignore=E,W spiral.py
	python spiral.py

pdf: fit
	a2ps spiral.py -o spiral.ps --pro=color
	ps2pdf spiral.ps
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite \
		-sOutputFile=Assg2.pdf \
		spiral.pdf spiral_class.pdf