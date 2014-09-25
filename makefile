all: extract_features.o learn classify

learn: src/Learn.cc src/NB.cc src/NB.h
	g++ $^ -o ./$@ -g

extract_features.o: src/extract_features.cc src/extract_features.h
	g++ $^ -o lib/$@ -g

classify: src/Classify.cc src/NB.cc src/NB.h
	g++ $^ -o ./$@ -g
