import sys
import os

# （已移除 sys.path.insert hack，改用 PYTHONPATH 管理模块路径）

from pathlib import Path
from assembled_ask.util.metric_switch_case import msc
from assembled_ask.util.metatask_base import get_metatask
from assembled_ask.ask_assembler import AskAssembler

if __name__ == "__main__":
    # -- Get Input Parameter
    openml_task_id = sys.argv[1]
    time_limit = int(sys.argv[2])
    memory_limit = int(sys.argv[3])
    folds_to_run = (
        [int(x) for x in sys.argv[4].split(",")]
        if "," in sys.argv[4] else [int(sys.argv[4])]
    )
    metric_name = sys.argv[5]
    base_folder_name = sys.argv[6]

    # -- Build paths (project root -> benchmark output directory)
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    tmp_output_dir = file_path.parent.parent / f"benchmark/output/{base_folder_name}/task_{openml_task_id}"
    print(f"Full Path Used: {tmp_output_dir}")

    # -- Get The Metatask
    print(f"Building Metatask for OpenML Task: {openml_task_id}")
    mt = get_metatask(openml_task_id)

    if openml_task_id == "-1":
        # If using a synthetic metatask (ID -1), adjust time limit (minutes to hours)
        time_limit = time_limit / 60

    metric_to_optimize = msc(metric_name, len(mt.class_labels) == 2, list(range(mt.n_classes)))

    # -- Init and run assembler (Auto-Sklearn)
    print("Run Assembler")
    assembler = AskAssembler(mt, tmp_output_dir, folds_to_run=folds_to_run)
    assembler.run_ask(metric_to_optimize, time_limit, memory_limit)

    print("Finished Run, Save State")
    for fold in folds_to_run:
        s_path = file_path.parent.parent / f"benchmark/state/{base_folder_name}/task_{openml_task_id}/"
        s_path.mkdir(parents=True, exist_ok=True)
        # Create a .done file per fold to mark completion
        (s_path / f"run_ask_on_metatask_{fold}.done").touch()
