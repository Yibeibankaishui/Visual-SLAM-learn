# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/parallels/vslam_learn/slam-homework/p3/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/parallels/vslam_learn/slam-homework/p3/code/build

# Include any dependencies generated for this target.
include CMakeFiles/draw_trajectory.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/draw_trajectory.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/draw_trajectory.dir/flags.make

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o: CMakeFiles/draw_trajectory.dir/flags.make
CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o: ../draw_trajectory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/parallels/vslam_learn/slam-homework/p3/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o -c /home/parallels/vslam_learn/slam-homework/p3/code/draw_trajectory.cpp

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/parallels/vslam_learn/slam-homework/p3/code/draw_trajectory.cpp > CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.i

CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/parallels/vslam_learn/slam-homework/p3/code/draw_trajectory.cpp -o CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.s

# Object files for target draw_trajectory
draw_trajectory_OBJECTS = \
"CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o"

# External object files for target draw_trajectory
draw_trajectory_EXTERNAL_OBJECTS =

draw_trajectory: CMakeFiles/draw_trajectory.dir/draw_trajectory.cpp.o
draw_trajectory: CMakeFiles/draw_trajectory.dir/build.make
draw_trajectory: /usr/local/lib/libpangolin.so
draw_trajectory: /usr/lib/aarch64-linux-gnu/libGL.so
draw_trajectory: /usr/lib/aarch64-linux-gnu/libGLU.so
draw_trajectory: /usr/lib/aarch64-linux-gnu/libGLEW.so
draw_trajectory: /usr/lib/aarch64-linux-gnu/libX11.so
draw_trajectory: CMakeFiles/draw_trajectory.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/parallels/vslam_learn/slam-homework/p3/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable draw_trajectory"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/draw_trajectory.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/draw_trajectory.dir/build: draw_trajectory

.PHONY : CMakeFiles/draw_trajectory.dir/build

CMakeFiles/draw_trajectory.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/draw_trajectory.dir/cmake_clean.cmake
.PHONY : CMakeFiles/draw_trajectory.dir/clean

CMakeFiles/draw_trajectory.dir/depend:
	cd /home/parallels/vslam_learn/slam-homework/p3/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/parallels/vslam_learn/slam-homework/p3/code /home/parallels/vslam_learn/slam-homework/p3/code /home/parallels/vslam_learn/slam-homework/p3/code/build /home/parallels/vslam_learn/slam-homework/p3/code/build /home/parallels/vslam_learn/slam-homework/p3/code/build/CMakeFiles/draw_trajectory.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/draw_trajectory.dir/depend

