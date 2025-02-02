BUILD_DIR      = cmake/build
BUILD_TEST_DIR = cmake/build_test
BIN_DIR        = bin
EXECUTABLE     = main

# Default build type
CMAKE_BUILD_TYPE = Debug

.PHONY: all
all: $(BUILD_DIR)/Makefile
	cd $(BUILD_DIR) && cmake --build .

$(BUILD_DIR)/Makefile:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) ../..

.PHONY: fast
fast:
	$(MAKE) CMAKE_BUILD_TYPE=Release all

.PHONY: run
run: all
	./$(BIN_DIR)/$(EXECUTABLE)

.PHONY: test
test: $(BUILD_TEST_DIR)/Makefile
	cd $(BUILD_TEST_DIR) && cmake --build .
	cd $(BUILD_TEST_DIR) && ctest --output-on-failure

$(BUILD_TEST_DIR)/Makefile:
	mkdir -p $(BUILD_TEST_DIR)
	cd $(BUILD_TEST_DIR) && cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) ../..

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BUILD_TEST_DIR) $(BIN_DIR)

.PHONY: rebuild
rebuild: clean all

.PHONY: rebuild_fast
rebuild_fast: clean fast

