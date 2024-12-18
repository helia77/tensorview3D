#Specify the version being used as well as the language
cmake_minimum_required(VERSION 3.29)
cmake_policy(VERSION 3.29)

if(POLICY CMP0167)
   cmake_policy(SET CMP0167 OLD)
endif()

#Name your project here
project(tensor3d LANGUAGES CXX CUDA)


#set the module directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")
set(CMAKE_CXX_STANDARD 20)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED core glfw-binding opengl3-binding)
find_package(GLEW REQUIRED)
find_package(TIRA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
#find_package(Boost CONFIG)
find_package(PNG REQUIRED)


#build the executable in the binary directory on MS Visual Studio
if ( MSVC )
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
	SET( LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_SCL_SECURE_NO_WARNINGS)
	add_definitions(-D_USE_MATH_DEFINES)
else()
	find_package(X11 REQUIRED)
	set(CMAKE_CXX_FLAGS "-Wall -Wextra")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif ( MSVC )


configure_file(Roboto-Medium.ttf Roboto-Medium.ttf COPYONLY)

# copy python scripts
configure_file(volume2tensor.py 
				volume2tensor.py COPYONLY)
				
# set the include directories
include_directories(
			${CMAKE_CURRENT_BINARY_DIR}
			${CMAKE_CURRENT_SOURCE_DIR}
			${GLFW_INCLUDE_DIRS}
			${TIRA_INCLUDE_DIRS}
			${X11_INCLUDE_DIR}
)

# tensorview executable
add_executable(tensorview3
				tensorview3.cpp
				cuda_calls.cu
				tensorvote.cu
				ImGuiFileBrowser/ImGuiFileBrowser.cpp
				ImGuiFileBrowser/ImGuiFileBrowser.h
				ImGuiFileBrowser/dirent.h
)


target_link_libraries(tensorview3
				PRIVATE glm::glm
				PRIVATE glfw
				PRIVATE GLEW::GLEW
				${OPENGL_LIBRARIES}
				${CMAKE_DL_LIBS}
				Boost::program_options
				PRIVATE imgui::imgui
)