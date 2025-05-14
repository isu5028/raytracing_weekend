FLAG = -Wall -O3 -std=c++14 -Wno-overloaded-virtual
FIG = 1_2 1_5 1_7 1_9 1_11 1_13 1_16 1_17 1_17_ex 1_19 1_20 1_22 1_24 1_27 1_28 1_29 1_31 1_34 1_35 1_38 1_39 2_1 2_1_ex1 2_1_ex2 2_6 2_7 2_11 2_12 2_13 2_14 2_15 2_16 2_17 2_19 2_21 2_22 2_23 2_24 2_27 2_29 2_30 3_6 3_6_ex1 3_6_ex2 3_9 3_11 3_12 3_13 3_14 3_15 3_16 3_17 3_18 3_19
LISTING = 3_1 3_2 3_3 3_4 3_5 3_6 3_7 3_9 3_15 3_16 3_17

EXE = $(FIG:%=fig_%.exe) $(LISTING:%=listing_%.exe)
IMAGE = $(FIG:%=fig_%.png) $(FIG:%=fig_%.ppm)

default: $(EXE)

image: $(IMAGE)

%.png: %.ppm
	magick $^ $@

%.ppm: %.exe
	$^ > $@

%.exe: %.cpp
	clang++ $(FLAG) $^ -O3 -o $@

.PHONY: clean
clean:
	rm -rf $(EXE) $(IMAGE)
