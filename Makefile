main: main.o
	g++ -o main main.o

main.o: main.cpp
	g++ -c main.cpp -std=c++14

clean:
	rm main *.o

main2: newton.cpp
	g++ -o main2 newton.cpp -std=c++14