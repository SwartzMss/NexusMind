# Mixed-language Runtime

系统通过 durable checkpoint 保存工作流状态，并在进程重启后 resume
execution。Hybrid retrieval 使用 reciprocal rank fusion 合并 lexical 与
semantic ranking，但不重新解释后端分数。
