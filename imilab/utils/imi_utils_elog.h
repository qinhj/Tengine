// ============================================================
//                        Easy Log APIs
// ------------------------------------------------------------
// Author:  qinhj@lsec.cc.ac.cn     Date:   2020/10/21
// ------------------------------------------------------------
// @Note:   Currently, the output is hard coded to stdout.
// ============================================================

#ifndef __IMI_UTILS_ELOG_H__
#define __IMI_UTILS_ELOG_H__

// user setting: default output
#ifndef FP_DFLT
#define FP_DFLT stdout
#endif /* !FP_DFLT */

#ifdef _MSC_VER

#include <Windows.h>// need for: STD_OUTPUT_HANDLE, ...
// 0: black; 1: blue; 2: green; 3: shallow green; 4: red
// 5: purple; 6: yellow; 7: white; 8: gray; 9: light blue
// ... F: high white
#define set_console_color(color) \
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color)

#define COLOR_RED       0x0C    // FOREGROUND_RED
#define COLOR_WHITE     0x0F
#define COLOR_YELLOW    0x06
#define COLOR_BLUE      0x03    // light blue

#else  /* !_MSC_VER */

#include <stdio.h>  // need for: fprintf
#define set_console_color(color) \
    fprintf(FP_DFLT, "\033[%dm", color)

#define COLOR_RED       31
#define COLOR_WHITE     0   // 37
#define COLOR_YELLOW    33
#define COLOR_BLUE      36  // light blue

#endif /* _MSC_VER */


#define log_printf(color_, _color, tag, ...) do {                       \
    set_console_color(color_);                                          \
    fprintf(FP_DFLT, tag "[line: %d][%s] ", __LINE__, __FUNCTION__);    \
    fprintf(FP_DFLT, ##__VA_ARGS__);                                    \
    set_console_color(_color);                                          \
} while (0)
#define log_error(...)  log_printf(COLOR_RED, COLOR_WHITE, "[E] ", ##__VA_ARGS__)
#define log_warn(...)   log_printf(COLOR_YELLOW, COLOR_WHITE, "[W] ", ##__VA_ARGS__)
#define log_info(...)   log_printf(COLOR_WHITE, COLOR_WHITE, "[I] ", ##__VA_ARGS__)
#define log_debug(...)  log_printf(COLOR_WHITE, COLOR_WHITE, "[D] ", ##__VA_ARGS__)

#if defined(DEBUG) || defined(_DEBUG) // || 1
#define log_verbose(...) log_printf(COLOR_WHITE, COLOR_WHITE, "[V] ", ##__VA_ARGS__)
#else /* !DEBUG && !_DEBUG */
#define log_verbose(...)
#endif /* DEBUG || _DEBUG */

#define log_echo(...)   fprintf(FP_DFLT, ##__VA_ARGS__)

#endif /* __IMI_UTILS_ELOG_H__ */
