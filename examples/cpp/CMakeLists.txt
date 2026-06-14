#-------------------------------------------------------------------------------
# C++ examples
#-------------------------------------------------------------------------------

find_package(OpenCV REQUIRED)

function(add_trtdetr_example target source_dir)
    add_executable(${target} "${source_dir}/${target}.cpp")

    target_include_directories(${target} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/modules/trtdetr/infer
    )

    target_link_libraries(${target} PRIVATE
        ${OpenCV_LIBS}
        trtdetr
    )

    set_target_compile_options(${target})

    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${target}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/examples/cpp/bin"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/examples/cpp/bin"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/examples/cpp/bin"
    )
endfunction()

if(BUILD_EXAMPLE_DETECT)
    add_trtdetr_example(detect "${CMAKE_CURRENT_SOURCE_DIR}/detect")
endif()

if(BUILD_EXAMPLE_SEGMENT)
    add_trtdetr_example(segment "${CMAKE_CURRENT_SOURCE_DIR}/segment")
endif()

if(BUILD_EXAMPLE_POSE)
    add_trtdetr_example(pose "${CMAKE_CURRENT_SOURCE_DIR}/pose")
endif()

if(BUILD_EXAMPLE_MULTI_THREAD)
    add_trtdetr_example(mutli_thread "${CMAKE_CURRENT_SOURCE_DIR}/mutli_thread")
endif()
