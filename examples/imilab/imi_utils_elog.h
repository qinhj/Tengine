// ============================================================
//                  Imilab Utils: Easy Log APIs
// ------------------------------------------------------------
// Author:  qinhongjie@imilab.com       Date:   2021/05/10
// ============================================================

#ifndef __IMI_UTILS_ELOG_H__
#define __IMI_UTILS_ELOG_H__

#if defined(DEBUG) || defined(_DEBUG) // || 1
#define log_debug(...)  printf(__VA_ARGS__)
#else // !DEBUG
#define log_debug(...)
#endif // DEBUG

#endif // !__IMI_UTILS_ELOG_H__
