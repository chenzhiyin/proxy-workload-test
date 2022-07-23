cmake_minimum_required(VERSION 2.8.12)
project(proxy_workload_test)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

link_directories(
	${PROJECT_SOURCE_DIR}/third_party/lib)

include_directories(
    /home/zhiyin/projects/.venv/workload-proxy/lib64/python3.6/site-packages/tensorflow_core/include/
	/home/zhiyin/projects/.venv/workload-proxy/lib/python3.6/site-packages/tensorflow/include/
	/home/zhiyin/package/rapidjson/include
	${PROJECT_SOURCE_DIR}/third_party/include)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

set(DEEP_CTR_LINKER_LIBS "")
list(APPEND DEEP_CTR_LINKER_LIBS
	/home/zhiyin/projects/proxy-workload-test/third_party/lib/libtensorflow_framework.so.2
	/home/zhiyin/projects/proxy-workload-test/third_party/lib/libtensorflow_cc.so
	m z pthread)

set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -g")
# executable
set(DEEP_MODEL "test.bin")
# add_executable(${DEEP_MODEL} "proxy_workload_test.cc")ll 
add_executable(${DEEP_MODEL} "parser_utility.cc" "proxy_workload_test.cc")
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
target_link_libraries(${DEEP_MODEL} ${DEEP_CTR_LINKER_LIBS})
