from dataclasses import dataclass

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
    train_csv_file: str = 'data/training_data_8192paths_3wps_1scale_EXPplanner_7seed.csv'
    val_csv_file: str = 'data/val_64paths_3wps_srand4.csv'
    test_csv_file: str = 'data/test_64paths_3wps_srand1.csv'
    best_model_path: str = 'models/endstate-selector_2024-06-08-22-10-02.pth'
    batch_size: int = 64
    input_size: int = 3*num_waypoints + 9
    output_size: int = num_vf_choices

    csv_input_col: int = 18 # upper bound is exlusive
    csv_label_col: int = 18