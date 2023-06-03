function(add_sanitizer_flags)
    if(NOT ENABLE_SANITIZE_ADDR AND NOT ENABLE_SANITIZE_UNDEF)
        return()
    endif()

    add_compile_options("-fno-omit-frame-pointer")
    add_link_options("-fno-omit-frame-pointer")

    if(ENABLE_SANITIZE_ADDR)
        add_compile_options("-fsanitize=address")
        add_link_options("-fsanitize=address")
    endif()

    if(ENABLE_SANITIZE_UNDEF)
        add_compile_options("-fsanitize=undefined")
        add_link_options("-fsanitize=undefined")
    endif()

    if(ENABLE_SANITIZE_LEAK)
        add_compile_options("-fsanitize=leak")
        add_link_options("-fsanitize=leak")
    endif()
endfunction(add_sanitizer_flags)