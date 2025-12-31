CXX          := gcc
#ARCH         := sm_75
CXXFLAGS     := -lOpenCL 
TARGET       := vetor_soma
SOURCE       := vetor_soma.c 

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
