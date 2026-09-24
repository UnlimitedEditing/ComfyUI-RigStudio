add_executable(rigstudio-a2f)

set_target_properties(rigstudio-a2f PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/bin
)

target_sources(rigstudio-a2f PRIVATE main.cpp)

target_link_libraries(rigstudio-a2f PRIVATE
    audio2x
    AudioFile::AudioFile
)
