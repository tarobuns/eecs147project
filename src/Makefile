# Define the CUDA compiler
NVCC := nvcc

# Define compiler flags
CFLAGS := -O2 -arch=sm_52 # Adjust the compute capability (e.g., sm_52) to match your GPU

# Define directories
SRC_DIR := .
DATA_DIR := ../data
BIN_DIR := ../bin

# Define the target executable
TARGET := $(BIN_DIR)/network_simulation

# Define source files
SRC_FILES := $(SRC_DIR)/main.cu $(SRC_DIR)/network.cu $(SRC_DIR)/network_kernel.cu $(SRC_DIR)/utils.cu

# Define object files
OBJ_FILES := $(SRC_FILES:.cu=.o)

# Define dependencies
DEPS := $(SRC_DIR)/network.h $(SRC_DIR)/utils.h

# Define rules
all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(CFLAGS) -o $@ $^

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	$(NVCC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

.PHONY: all clean
