"""
중앙 하이퍼파라미터 관리 설정 파일.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # 랜덤 시드
    "random_seed": 42,
    "device": "cuda",  # "cpu" 또는 "cuda"

    # 데이터 경로 (실제 폴더 구조 기준)
    "train_images_dir":    os.path.join(BASE_DIR, "data", "train", "images"),
    "train_labels_dir":    os.path.join(BASE_DIR, "data", "train", "1st_manual"),
    "train_masks_dir":     os.path.join(BASE_DIR, "data", "train", "masks"),
    "train_skeletons_dir": os.path.join(BASE_DIR, "data", "train", "skeletons"),

    "test_images_dir":     os.path.join(BASE_DIR, "data", "test", "images"),
    "test_masks_dir":      os.path.join(BASE_DIR, "data", "test", "masks"),
    "test_skeletons_dir":  os.path.join(BASE_DIR, "data", "test", "skeletons"),

    "checkpoints_dir":     os.path.join(BASE_DIR, "checkpoints"),
    "results_dir":         os.path.join(BASE_DIR, "results"),

    # 신규 데이터셋 (E 드라이브)
    "new_images_dir":    r"E:\Downloads\dataset\kaggl\vessel_tracking_dataset\vessel_test",
    "new_gt_dir":        r"E:\Downloads\dataset\kaggl\vessel_tracking_dataset\blood_vessel",
    "new_skeletons_dir": r"E:\Downloads\dataset\kaggl\vessel_tracking_dataset\skeletons",
    "full_fundus_dir":   r"E:\Downloads\dataset\kaggl\vessel_tracking_dataset\full_fundus",

    # 데이터 분할
    "val_ratio":    0.2,   # val 비율 (전체의 20% = ~96장)
    "val_interval": 500,   # 몇 에피소드마다 val 평가할지

    # 환경
    "patch_size": 31,
    "n_actions": 8,
    "max_steps": 1000,
    "max_off_vessel": 30,   # FOV 밖 연속 허용 스텝 (train/eval 통일)
    "near_skel_radius": 2,   # near_skel 팽창 반경 px (커널 크기 = 2r+1)

    # 보상
    "reward_on_vessel":        3.0,   # 스켈레톤 위 픽셀 정확히 밟았을 때
    "reward_near_vessel":      0.8,   # 스켈레톤 인접 위치
    "reward_off_vessel":      -1.0,   # 혈관과 무관한 위치
    "reward_step":             0.1,   # 매 스텝 이동 시 기본 보상 (이동 인센티브)
    "reward_revisit":         -3.0,   # 최근 local_revisit_window 스텝 내 재방문 패널티
    "reward_boundary":        -8.0,   # FOV 경계 이탈 패널티
    "reward_completion":      10.0,   # on_vessel 60% 이상 완료 보상
    "reward_coverage_penalty": 30.0,  # 에피소드 종료 후 미커버 스켈레톤 비율 × 이 값만큼 패널티
    "local_revisit_window":    20,    # 이 스텝 이내 재방문에만 패널티 적용 (오래된 재방문 허용)

    # DQN
    "learning_rate":    1e-4,
    "gamma":            0.99,
    "epsilon_start":    1.0,
    "epsilon_end":      0.05,
    "epsilon_decay":    0.99995,  # 탐색 충분히 보장
    "batch_size":       128,
    "replay_buffer":    50000,
    "target_update":    500,

    # 학습
    "n_episodes":    100000,
    "save_interval": 100,
    "log_interval":  50,
}
