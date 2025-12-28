CXX          := nvcc
ARCH         := sm_75
CXXFLAGS     := -std=c++17 -O3
TARGET       := cuda
SOURCE       := cuda.cu

$(TARGET): $(SOURCE)
	$(CXX) -arch=$(ARCH) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
