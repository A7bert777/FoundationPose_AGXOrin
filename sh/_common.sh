#!/usr/bin/env bash
# 公共辅助函数（由 setup.sh / build_ext.sh / download.sh source）
#
# run_logged —— 让长任务的输出「终端实时可见 + 日志落盘」
#
# 为什么需要它：
#   pip / gdown 只有在 stdout 是 tty 时才显示实时进度条。若直接写成
#       cmd >> "$LOG/xxx.log" 2>&1
#   终端会长时间一片空白（216MB 的 torch wheel 要下十几分钟，2.6G 的示例
#   数据更久），完全看不出是否卡住。
#   这里用 script(pty) 给子进程分配一个伪终端，从而两者兼得。
#
# 依赖：util-linux 的 script（Ubuntu/JetPack 默认自带）。
#       缺失时退化为 tee，进度条会变成单行输出，但仍是实时的。

run_logged() {   # 用法：run_logged <日志文件> <命令...>
  local log="$1"; shift
  mkdir -p "$(dirname "$log")"
  if command -v script >/dev/null 2>&1; then
    # -q 不打印 script 自己的提示，-e 透传子进程退出码，-a 追加写入
    script -q -e -a -c "$(printf '%q ' "$@")" "$log"
  else
    "$@" 2>&1 | tee -a "$log"
  fi
}
