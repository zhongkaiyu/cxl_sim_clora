cmake_minimum_required(VERSION 2.8.4)
project(main)

add_custom_target(main COMMAND make all
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)