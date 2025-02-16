all: clean build_first build_second

build_first: lab2first.c
	gcc -O3 lab2first.c -o lab2first -lm -fopenmp

build_second: lab2second.c
	gcc -O3 lab2second.c -o lab2second -lm -fopenmp

clean:
	rm -rf lab2first lab2second time.txt

test: clean build_first build_second tester.bash
	bash tester.bash