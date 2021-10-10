fit: agnewsclass.py
	black agnewsclass.py
	flake8 --ignore=E,W agnewsclass.py
	python agnewsclass.py > results.txt

pdf: fit
	a2ps agnewsclass.py -o agnewsclass.ps --pro=color
	a2ps results.txt -o results.ps --pro=color
	ps2pdf agnewsclass.ps
	ps2pdf results.ps
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite \
		-sOutputFile=Assg5.pdf \
		agnewsclass.pdf results.pdf