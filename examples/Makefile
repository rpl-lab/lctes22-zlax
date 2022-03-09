.DEFAULT_GOAL:=help

help:
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  make %s\t# %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

coin: coin.zls ## simulate the coin example
	zluciole -prob -n 20 -s main coin.zls

hmm: hmm.zls ## simulate the hmm example
	zluciole -prob -n 100 -s main hmm.zls

hmm-plot: hmm.zls ## simulate the hmm example with graphical interface using gnuplot
	zluciole -prob -n 100 -s main hmm.zls | tr -d '()' > hmm.data
	gnuplot -p -e "plot 'hmm.data' using 1:2 pointtype 7 pointsize 0.75 title 'obs',  '' using 1:3:4 with yerrorbars linestyle 3 title 'std', '' using 1:3 with lines linestyle 7 title 'mean'"

clean: ## cleanup
	rm -rf *.zci hmm.py coin.py hmm.data
