help:
	@echo "Help:"
	@echo "  make coin      # simulate the coin example"
	@echo "  make hmm       # simulate the hmm example"
	@echo "  make hmm-plot  # simulate the hmm example with graphical interface using gnuplot"

coin:
	zluciole -prob -n 20 -s main coin.zls

hmm:
	zluciole -prob -n 100 -s main hmm.zls

hmm-plot:
	zluciole -prob -n 100 -s main hmm.zls | tr -d '()' > hmm.data
	gnuplot -p -e "plot 'hmm.data' using 1:2 pointtype 7 pointsize 0.75 title 'obs',  '' using 1:3:4 with yerrorbars linestyle 3 title 'std', '' using 1:3 with lines linestyle 7 title 'mean'"

clean:
	rm -rf *.zci hmm.py coin.py hmm.data
