all: clean build_first build_second

build_first: lab1first.cpp
	mpicxx -O3 lab1first.cpp -o lab1first

build_second: lab1second.cpp
	mpicxx -O3 lab1second.cpp -o lab1second

test: test.bash
	bash test.bash

clean:
	rm -rf lab1first lab1second