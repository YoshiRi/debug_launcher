cmake_minimum_required(VERSION 3.8)
project(debug_launcher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(autoware_cmake REQUIRED)
autoware_package()

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  launch
  config
  rviz
)

# install python source
install(DIRECTORY src/
  DESTINATION lib/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py"
)

# install scripts
file(GLOB SCRIPTS_FILES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/scripts"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.sh")
foreach(SCRIPTS_FILE ${SCRIPTS_FILES})
  message(STATUS "Installing script: ${SCRIPTS_FILE}")
  install(PROGRAMS scripts/${SCRIPTS_FILE}
    DESTINATION lib/${PROJECT_NAME}
  )
endforeach()