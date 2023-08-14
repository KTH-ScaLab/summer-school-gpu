HIPCC=hipcc

SOURCES = vectoradd_hip.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./vectoradd_hip.exe

all: $(EXECUTABLE)

CXXFLAGS = --offload-arch=gfx90a

CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS) 
	$(HIPCC) $(OBJECTS) -o $@

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
