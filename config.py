from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Planner settings
    num_waypoints: int = 3

    num_vf_angles: int = 18
    num_vf_mags: int = 2
    num_vf_choices: int = (num_vf_angles)*(num_vf_mags)+1
    num_af_angles: int = 19
    num_af_mags: int = 2
    num_af_choices: int = (num_af_angles)*(num_af_mags)+1

    # Model settings
    train_csv_file: List[str] = field(default_factory=lambda: ['data/1scale/training_data_8192paths_3wps_1scale_EXPplanner_7seed.csv', 'data/1scale/training_data_8192paths_3wps_1scale_EXPplanner_9seed.csv'])
    val_csv_file: List[str] = field(default_factory=lambda: ['data/1scale/val_data_4096paths_3wps_1scale_EXPplanner_10seed.csv'])
    test_csv_file: List[str] = field(default_factory=lambda: ['data/1scale/test_data_1070paths_3wps_1scale_EXPplanner_10seed.csv'])
    best_model_path: str = 'models/endstate-selector_06-11-21-01.pth'
    batch_size: int = 64
    input_size: int = 3*num_waypoints + 9
    output_size: int = num_vf_choices

    csv_input_col: int = 18 # upper bound is exlusive
    vf_desc_col: int = 18
    af_desc_col: int = 19
    csv_label_col: int = vf_desc_col