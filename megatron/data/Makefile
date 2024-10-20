PYTHON_PATH=/lustre/orion/bif151/scratch/vaibhav_016/miniconda3/bin/python3
CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell $(PYTHON_PATH) -m pybind11 --includes)
LIBNAME = helpers
LIBEXT = $(shell $(PYTHON_PATH)-config --extension-suffix)

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
