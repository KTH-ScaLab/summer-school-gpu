HIPCC=hipcc

SOURCES = MatrixMultiplication.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./MatrixMultiplication

all: $(EXECUTABLE)

CXXFLAGS = --offload-arch=gfx90a
CXXFLAGS += -I./include

CXX=$(HIPCC)

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC)  $(OBJECTS) -o $@

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
