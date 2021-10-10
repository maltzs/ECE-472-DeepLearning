fit: cifarclass.py
	black cifarclass.py
	flake8 --ignore=E,W cifarclass.py
	python cifarclass.py > results10.txt
	python cifarclass.py --cifar100=True --dense_widths="1024" > results100.txt

pdf: fit
	a2ps cifarclass.py -o cifarclass.ps --pro=color
	a2ps results10.txt -o results10.ps --pro=color
	a2ps results100.txt -o results100.ps --pro=color
	ps2pdf cifarclass.ps
	ps2pdf results10.ps
	ps2pdf results100.ps
	gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite \
		-sOutputFile=Assg4.pdf \
		cifarclass.pdf results10.pdf results100.pdf