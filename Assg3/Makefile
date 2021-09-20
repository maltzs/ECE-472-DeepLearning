fit: mnistclass.py
	black mnistclass.py
	flake8 --ignore=E,W mnistclass.py
	python mnistclass.py > results.txt

pdf: fit
	a2ps mnistclass.py -o mnistclass.ps --pro=color
	a2ps results.txt -o results.ps --pro=color
	ps2pdf mnistclass.ps
	ps2pdf results.ps
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite \
		-sOutputFile=Assg3.pdf \
		mnistclass.pdf results.pdf