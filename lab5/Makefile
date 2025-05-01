all: clean build

build: main.c
	mpicc -O3 main.c -pthread -lm -o main

clean:
	rm -rf main