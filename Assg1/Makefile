fit: sinereg.py
	black sinereg.py
	flake8 --ignore=E,W sinereg.py
	python sinereg.py

pdf: fit
	a2ps sinereg.py -o sinereg.ps --pro=color
	ps2pdf sinereg.ps
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite \
		-sOutputFile=Assg1.pdf \
		sinereg.pdf fit.pdf