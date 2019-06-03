main: main.o
	g++ -o main main.o

main.o: main.cpp
	g++ -c main.cpp -std=c++14

clean:
	rm main *.o
