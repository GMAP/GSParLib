# Compilers
COMPILER := g++
# Directories
SRCDIR := src
BUILDDIR := build
TARGETDIR := bin
EXAMPLESDIR := examples
EXAMPLEDRIVERAPIDIR := $(EXAMPLESDIR)/driver_api
EXAMPLEPATTERNAPIDIR := $(EXAMPLESDIR)/pattern_api
EXAMPLESEQUENTIALDIR := $(EXAMPLESDIR)/sequential
THIRDPTDIR := thirdpt
MARX2DIR := $(THIRDPTDIR)/marX2
LIBMARX2PATH := $(MARX2DIR)/libmarX2.a
# Names
LIBNAME := gspar
GSPARNAME := gspar
CUDANAME := cuda
OCLNAME := opencl
DRIVERAPINAME := driverapi
PATTERNAPINAME := patternapi
SEQUENTIALNAME := seq
# App names
MANDELNAME := mandel
LANEDETECTIONNAME := lanedetection
# Target
TARGET := $(TARGETDIR)/lib$(LIBNAME).so
# Others
SPACE := 
SPACE +=
SRCEXT := cpp
EXAMPLESTARGETPREFIX := ex

# Files
SOURCES := $(wildcard $(SRCDIR)/*.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

CFLAGS := -Wall -std=c++14 -O3 -Wno-reorder
LIB := -Llib -L/usr/local/cuda/targets/x86_64-linux/lib -L/usr/local/cuda/targets/x86_64-linux/lib/stubs -L/usr/local/cuda/lib -L/usr/local/cuda/lib64
LIBOCL := -lOpenCL
LIBCUDADRIVER := -lcuda
LIBCUDANVRTC := -lnvrtc
LIBPTHREAD := -pthread
PATHSLIB := -I/usr/local/cuda/include -Isrc
PATHSTEST := $(PATHSLIB) -I$(THIRDPTDIR) -I$(EXAMPLESDIR)/include
TESTLIB := -L$(TARGETDIR) -l$(LIBNAME)

# Valid values for CL_TARGET_OPENCL_VERSION: 100, 110, 120, 200, 210, 220, 300
ifdef CL_TARGET_OPENCL_VERSION
	DEFSGPU +=-DCL_TARGET_OPENCL_VERSION=${CL_TARGET_OPENCL_VERSION}
else
	DEFSGPU +=-DCL_TARGET_OPENCL_VERSION=300
endif

PATHOPENCV := -I/usr/local/include/opencv4
LIBSOPENCV := -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
EXTRADEPS := 
INCMANDEL := 
LIBMANDEL := 
ifdef DEBUG
	DEFS +=-DDEBUG
	DEFSGPU +=-DGSPAR_DEBUG
	EXTRADEPS := $(LIBMARX2PATH)
	INCMANDEL := -I$(MARX2DIR) -L$(MARX2DIR)
	LIBMANDEL := -lmarX2 -lX11 -lm
endif

CLR_BLUE := \033[0;34m
CLR_ORANGE := \033[0;33m
CLR_DARKCYAN := \033[0;36m
CLR_NO := \033[0m

# Functions
get_paths_mandel = $(if $(findstring $(MANDELNAME), $(1)), $(INCMANDEL))
get_paths_lanedetection = $(if $(findstring $(LANEDETECTIONNAME), $(1)), $(PATHOPENCV))
get_paths = $(strip $(PATHSTEST) $(LIB) $(call get_paths_mandel, $(1)) $(call get_paths_lanedetection, $(1)) )

get_libs_mandel = $(if $(findstring $(MANDELNAME), $(1)), $(LIBMANDEL))
get_libs_lanedetection = $(if $(findstring $(LANEDETECTIONNAME), $(1)), $(LIBSOPENCV))
get_libs = $(strip $(call get_libs_mandel, $(1)) $(call get_libs_lanedetection, $(1)))

# Driver API examples
EXAMPLESOURCES_DRIVERAPI := $(wildcard $(EXAMPLEDRIVERAPIDIR)/*.$(SRCEXT))
EXAMPLETARGETS_DRIVERAPI_CUDA := $(patsubst $(EXAMPLEDRIVERAPIDIR)/%,$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(DRIVERAPINAME)_%,$(EXAMPLESOURCES_DRIVERAPI:.$(SRCEXT)=_$(CUDANAME)))
EXAMPLETARGETS_DRIVERAPI_OPENCL := $(patsubst $(EXAMPLEDRIVERAPIDIR)/%,$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(DRIVERAPINAME)_%,$(EXAMPLESOURCES_DRIVERAPI:.$(SRCEXT)=_$(OCLNAME)))
# Pattern API examples
EXAMPLESOURCES_PATTERNAPI := $(wildcard $(EXAMPLEPATTERNAPIDIR)/*.$(SRCEXT))
EXAMPLETARGETS_PATTERNAPI_CUDA := $(patsubst $(EXAMPLEPATTERNAPIDIR)/%,$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(PATTERNAPINAME)_%,$(EXAMPLESOURCES_PATTERNAPI:.$(SRCEXT)=_$(CUDANAME)))
EXAMPLETARGETS_PATTERNAPI_OPENCL := $(patsubst $(EXAMPLEPATTERNAPIDIR)/%,$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(PATTERNAPINAME)_%,$(EXAMPLESOURCES_PATTERNAPI:.$(SRCEXT)=_$(OCLNAME)))
# Sequential examples
EXAMPLESOURCES_SEQUENTIAL := $(wildcard $(EXAMPLESEQUENTIALDIR)/*.$(SRCEXT))
EXAMPLETARGETS_SEQUENTIAL := $(patsubst $(EXAMPLESEQUENTIALDIR)/%,$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(SEQUENTIALNAME)_%,$(EXAMPLESOURCES_SEQUENTIAL:.$(SRCEXT)=))


# Build targets

$(TARGET): $(OBJECTS) | $(TARGETDIR)
	@echo "${CLR_DARKCYAN}Linking dynamic library ${CLR_ORANGE}$(TARGET)${CLR_NO}..."
	$(COMPILER) $(DEFS) $(DEFSGPU) -shared -fPIC -o $(TARGET) $^ $(LIB) $(LIBOCL) $(LIBCUDADRIVER) $(LIBCUDANVRTC)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT) | $(BUILDDIR)
	@echo "${CLR_DARKCYAN}Compiling and assembling object ${CLR_ORANGE}$@${CLR_NO}..."
	$(COMPILER) $(DEFS) $(DEFSGPU) $(CFLAGS) $(PATHSLIB) -c -fPIC -o $@ $<

$(TARGETDIR):
	@mkdir -p $@

$(BUILDDIR):
	@mkdir -p $@


.PHONY: examples
examples: examples_driver_api examples_pattern_api examples_sequential

# Driver API examples
examples_driver_api: $(EXAMPLETARGETS_DRIVERAPI_CUDA) $(EXAMPLETARGETS_DRIVERAPI_OPENCL)
$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(DRIVERAPINAME)_%: $(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(DRIVERAPINAME)_%_$(CUDANAME) $(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(DRIVERAPINAME)_%_$(OCLNAME) ;
# Lib to CUDA
$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(DRIVERAPINAME)_%_$(CUDANAME): $(EXAMPLEDRIVERAPIDIR)/%.$(SRCEXT) $(TARGET) $(EXTRADEPS) | $(TARGETDIR)
	@echo "${CLR_DARKCYAN}Building GSPar Driver API example ${CLR_ORANGE}$@${CLR_DARKCYAN} from $<${CLR_NO}"
	$(COMPILER) $(DEFS) $(DEFSGPU) -DGSPARDRIVER_CUDA $(CFLAGS) $< $(call get_paths, $<) $(TESTLIB) -o $@ $(LIBPTHREAD) $(call get_libs, $<)
# Lib to OpenCL
$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(DRIVERAPINAME)_%_$(OCLNAME): $(EXAMPLEDRIVERAPIDIR)/%.$(SRCEXT) $(TARGET) $(EXTRADEPS) | $(TARGETDIR)
	@echo "${CLR_DARKCYAN}Building GSPar Driver API example ${CLR_ORANGE}$@${CLR_DARKCYAN} from $<${CLR_NO}"
	$(COMPILER) $(DEFS) $(DEFSGPU) -DGSPARDRIVER_OPENCL $(CFLAGS) $< $(call get_paths, $<) $(TESTLIB) -o $@ $(LIBPTHREAD) $(call get_libs, $<)

# Pattern API examples
examples_pattern_api: $(EXAMPLETARGETS_PATTERNAPI_CUDA) $(EXAMPLETARGETS_PATTERNAPI_OPENCL)
$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(PATTERNAPINAME)_%: $(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(PATTERNAPINAME)_%_$(CUDANAME) $(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(PATTERNAPINAME)_%_$(OCLNAME) ;
# Lib to CUDA
$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(PATTERNAPINAME)_%_$(CUDANAME): $(EXAMPLEPATTERNAPIDIR)/%.$(SRCEXT) $(TARGET) $(EXTRADEPS) | $(TARGETDIR)
	@echo "${CLR_DARKCYAN}Building GSPar Pattern API example ${CLR_ORANGE}$@${CLR_DARKCYAN} from $<${CLR_NO}"
	$(COMPILER) $(DEFS) $(DEFSGPU) -DGSPARDRIVER_CUDA $(CFLAGS) $< $(call get_paths, $<) $(TESTLIB) -o $@ $(LIBPTHREAD) $(call get_libs, $<)
# Lib to OpenCL
$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(PATTERNAPINAME)_%_$(OCLNAME): $(EXAMPLEPATTERNAPIDIR)/%.$(SRCEXT) $(TARGET) $(EXTRADEPS) | $(TARGETDIR)
	@echo "${CLR_DARKCYAN}Building GSPar Pattern API example ${CLR_ORANGE}$@${CLR_DARKCYAN} from $<${CLR_NO}"
	$(COMPILER) $(DEFS) $(DEFSGPU) -DGSPARDRIVER_OPENCL $(CFLAGS) $< $(call get_paths, $<) $(TESTLIB) -o $@ $(LIBPTHREAD) $(call get_libs, $<)

# Sequential examples
examples_sequential: $(EXAMPLETARGETS_SEQUENTIAL)
$(TARGETDIR)/$(EXAMPLESTARGETPREFIX)_$(SEQUENTIALNAME)_%: $(EXAMPLESEQUENTIALDIR)/%.$(SRCEXT) | $(TARGETDIR)
	@echo "${CLR_DARKCYAN}Building sequential example ${CLR_ORANGE}$@${CLR_DARKCYAN} from $<${CLR_NO}"
	$(COMPILER) $(DEFS) $(CFLAGS) $< $(call get_paths, $<) -o $@ $(call get_libs, $<)

$(LIBMARX2PATH): $(MARX2DIR)/marX2.c $(MARX2DIR)/marX2.h
	@echo "${CLR_DARKCYAN}Building ${CLR_ORANGE}$(LIBMARX2PATH)${CLR_DARKCYAN}${CLR_NO}"
	gcc -c -Wall -O3 -I/usr/X11R6/include -I$(MARX2DIR) $(MARX2DIR)/marX2.c -o $(MARX2DIR)/marX2.o
	ar -rv $(LIBMARX2PATH) $(MARX2DIR)/marX2.o
	ranlib $(LIBMARX2PATH)

.PHONY: clean
clean:
	@echo "${CLR_DARKCYAN}Cleaning...${CLR_NO}"; 
	$(RM) -r $(BUILDDIR) $(TARGETDIR) $(MARX2DIR)/*.a $(MARX2DIR)/*.o

clean_lib:
	@echo "${CLR_DARKCYAN}Cleaning lib $(TARGET)...${CLR_NO}"; 
	$(RM) $(OBJECTS) $(TARGET)

all: $(TARGET) examples
