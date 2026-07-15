# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hongpei Li

# python_bindings/CMakeLists.txt
#
# Tiny pybind11 wrapper around the CARDAL C ABI. Only activated when the
# top-level configure is called with -DCARDAL_BUILD_PYTHON=ON (typically
# via scikit-build-core / pip install).

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE _core.cpp)

target_link_libraries(_core PRIVATE sdp_core)
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# CUDA symbols get pulled in transitively via sdp_core. Force position-
# independent code so the .so links cleanly.
set_target_properties(_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# When loaded from python/cardal/_core.*.so, the RPATH should reach
# neighboring shared libs bundled in the wheel (if any).
set_target_properties(_core PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# --- Install the compiled module into python/cardal/ ---
# scikit-build-core copies whatever we install into the wheel's package tree.
install(TARGETS _core
        LIBRARY DESTINATION cardal
        COMPONENT python)

# --- Auto-generate .pyi stub via pybind11-stubgen (POST_BUILD, best-effort) ---
find_program(PYBIND11_STUBGEN pybind11-stubgen)
if(PYBIND11_STUBGEN)
    add_custom_command(TARGET _core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env
                PYTHONPATH=$<TARGET_FILE_DIR:_core>
                ${PYBIND11_STUBGEN}
                    --output-dir ${CMAKE_CURRENT_BINARY_DIR}
                    --ignore-invalid-expressions ".*"
                    _core
        COMMENT "Generating .pyi stub for cardal._core"
        VERBATIM
    )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_core.pyi
            DESTINATION cardal
            COMPONENT python
            OPTIONAL)
endif()
