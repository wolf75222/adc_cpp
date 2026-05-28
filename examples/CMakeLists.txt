add_executable(diocotron diocotron.cpp)
target_link_libraries(diocotron PRIVATE adc::adc)

add_executable(diocotron_column diocotron_column.cpp)
target_link_libraries(diocotron_column PRIVATE adc::adc)
