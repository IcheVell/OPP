all: clean build_first build_second

build_first: lab4.cpp
	mpicxx -O3 lab4.cpp -o lab4

build_second: lab4rhoonce.cpp
	mpicxx -O3 lab4rhoonce.cpp -o lab4rhoonce

clean:
	rm -rf lab4 lab4rhoonce