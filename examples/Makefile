all:
	@printf "Choose one of coin, hmm, or hmm-plot.\n"

coin:
	zluciole -prob -n 20 -s main coin.zls

hmm:
	zluciole -prob -n 100 -s main hmm.zls

hmm-plot:
	zluciole -prob -n 100 -s main hmm.zls | tr -d '()' > hmm.data
	gnuplot -p -e "plot 'hmm.data' using 1:2, '' using 1:3:4 with yerrorbars"
