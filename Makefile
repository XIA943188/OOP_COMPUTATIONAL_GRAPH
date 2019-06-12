main1: main1.o
	g++ -o main1 main1.o

main1.o: main1.cpp
	g++ -c main1.cpp -std=c++14

main2: main2.o
	g++ -o main2 main2.o

main2.o: main2.cpp
	g++ -c main2.cpp -std=c++14

main3: newton.cpp
	g++ -o main3 newton.cpp -std=c++14
	
main4: main4.o
	g++ -o main4 main4.o

main4.o: main4.cpp
	g++ -c main4.cpp -std=c++14

clean:
	rm main1 main2 main3 main4 *.o