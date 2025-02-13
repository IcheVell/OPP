all: clean build

build: lab2.c
	gcc -O3 lab2.c -o lab2 -lm -fopenmp

clean:
	rm -rf lab2 time.txt

test: clean build run.bash
	bash run.bash