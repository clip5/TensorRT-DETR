#-------------------------------------------------------------------------------
# 收集源文件并添加到目标
#-------------------------------------------------------------------------------
function(add_target_compile_files target)
    # 添加项目源码目录到目标的私有包含目录
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # 收集源文件（使用 CONFIGURE_DEPENDS 自动更新）
    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cu"
    )

    # 将收集到的源文件添加到目标
    target_sources(${target} PRIVATE ${SOURCES})
endfunction()

#-------------------------------------------------------------------------------
# 配置目标的通用属性
#-------------------------------------------------------------------------------
function(configure_target_common_properties target)
    # 设置目标的通用属性
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "${target}"
        CXX_VISIBILITY_PRESET "hidden"
    )
    # 配置 CUDA 和 TensorRT 相关属性
    configure_cuda_trt_target(${target})
    # 设置目标的编译选项
    set_target_compile_options(${target})
endfunction()

#-------------------------------------------------------------------------------
# 安装目标
#-------------------------------------------------------------------------------
function(install_target target target_type)
    if(target_type STREQUAL "library")
        # 安装库目标
        install(TARGETS ${target}
            EXPORT ${target}Targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            # ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            # RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    elseif(target_type STREQUAL "python")
        # 安装 Python 绑定目标
        install(TARGETS ${target}
            EXPORT ${target}Targets
            LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/trtdetr/libs
        )
    else()
        message(WARNING "Unknown target type: ${target_type}")
    endif()
endfunction()

#-------------------------------------------------------------------------------
# trtdetr 库
#-------------------------------------------------------------------------------
add_library(trtdetr SHARED)
add_target_compile_files(trtdetr)
configure_target_common_properties(trtdetr)
install_target(trtdetr "library")

# 设置变量
set(PACKAGE_VERSION "${PROJECT_VERSION}")
set(PACKAGE_DIRS "${CMAKE_INSTALL_PREFIX}")
set(PACKAGE_INCLUDE_DIRS "${CMAKE_INSTALL_INCLUDEDIR}")
set(PACKAGE_LIBRARY_DIRS "${CMAKE_INSTALL_LIBDIR}")
set(PACKAGE_LIBRARIES "trtdetr")

# 安装头文件
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/infer/trtdetr.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# 创建 tensorrt-detr-config.cmake 文件
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/tensorrt-detr-config.cmake.in"
    "${PROJECT_SOURCE_DIR}/cmake/tensorrt-detr-config.cmake"
    @ONLY
)
install(FILES "${PROJECT_SOURCE_DIR}/cmake/tensorrt-detr-config.cmake"
    DESTINATION "${CMAKE_INSTALL_PREFIX}"
)

#-------------------------------------------------------------------------------
# Python 绑定（可选）
#-------------------------------------------------------------------------------
if(BUILD_PYTHON)
    # 创建 Python 绑定模块
    pybind11_add_module(py_trtdetr ${CMAKE_CURRENT_SOURCE_DIR}/binding/trtdetr.cpp)
    add_target_compile_files(py_trtdetr)
    configure_target_common_properties(py_trtdetr)
    set_target_properties(py_trtdetr PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/trtdetr/libs"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/trtdetr/libs"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/trtdetr/libs"
    )
    # install_target(py_trtdetr "python")

    # 配置 Python 配置文件
    configure_file(
        ${PROJECT_SOURCE_DIR}/trtdetr/c_lib_wrap.py.in
        ${PROJECT_SOURCE_DIR}/trtdetr/c_lib_wrap.py
    )

    add_custom_target(python_wheel ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/dist
        COMMAND ${Python_EXECUTABLE} -m pip wheel --no-deps --wheel-dir ${PROJECT_SOURCE_DIR}/dist .
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        DEPENDS py_trtdetr
        COMMENT "Building Python wheel into ${PROJECT_SOURCE_DIR}/dist"
    )
endif()