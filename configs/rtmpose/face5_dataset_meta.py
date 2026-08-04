dataset_info = dict(
    dataset_name='face5',
    paper_info=dict(
        author='local',
        title='face_pose_drive_v1 five-point face dataset',
        container='local',
        year='2026',
        homepage=''),
    keypoint_info={
        0: dict(
            name='left_eye',
            id=0,
            color=[255, 80, 80],
            type='upper',
            swap='right_eye'),
        1: dict(
            name='right_eye',
            id=1,
            color=[80, 160, 255],
            type='upper',
            swap='left_eye'),
        2: dict(
            name='nose_tip',
            id=2,
            color=[255, 220, 0],
            type='upper',
            swap=''),
        3: dict(
            name='left_mouth_corner',
            id=3,
            color=[255, 140, 0],
            type='upper',
            swap='right_mouth_corner'),
        4: dict(
            name='right_mouth_corner',
            id=4,
            color=[160, 80, 255],
            type='upper',
            swap='left_mouth_corner'),
    },
    skeleton_info={
        0: dict(
            link=('left_eye', 'nose_tip'), id=0, color=[255, 220, 0]),
        1: dict(
            link=('right_eye', 'nose_tip'), id=1, color=[255, 220, 0]),
        2: dict(
            link=('nose_tip', 'left_mouth_corner'),
            id=2,
            color=[255, 140, 0]),
        3: dict(
            link=('nose_tip', 'right_mouth_corner'),
            id=3,
            color=[160, 80, 255]),
    },
    joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    sigmas=[0.025, 0.025, 0.026, 0.035, 0.035])
