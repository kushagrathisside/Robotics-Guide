YT = "https://www.youtube.com/watch?v="
NANDI = {1: "7_EEBOgdGM8", 2: "4eAlCjFMq2Q", 3: "2ww-3c08BIs", 6: "WHIlBbJtdxk", 7: "588iwEM-Dto",
         8: "HmveMBMFTtU", 9: "MCN5ST40cy8", 10: "eItbJ-WYzAE", 11: "m50tWpx-t0I",
         12: "wjaFmQKnLOA", 13: "UxC8Ho5e6_Y", 14: "uGzl-TW68Pw", 15: "CJdCTyDpX4U"}
NANDI_TITLES = {6: "Basics of spatial transformation", 7: "Basics of rotational transformation",
                8: "How to represent orientation?", 9: "Orientation description of a rigid body",
                10: "Rotation matrix when the body is rotated about a given axis",
                11: "Gimbal lock explained in a practical manner",
                12: "Typical solution of Euler angles, type II",
                13: "Orientation description using quaternion: why is it more useful?",
                14: "Use of quaternion for rotation",
                15: "Spatial transformation explained in a simple way"}
TE = {"spatial": ("4Y1_y9DI_Hw", "Spatial descriptions and transformation matrices"),
      "spatial_ex": ("r244ADkuexk", "Spatial descriptions — worked example"),
      "compound": ("sm7d1A2npWA", "Compound transformation matrices and inverse transformation matrices"),
      "euler": ("uG0P9RX65Ek", "Fixed and Euler angle representation"),
      "euler_ex": ("8FD1jgb3Df4", "Euler angles — worked example")}
MR = {"3": ("29LhXWjn7Pc", "Chapter 3 introduction: rigid-body motions"),
      "3.2.1a": ("OZucG1DY_sY", "3.2.1 Rotation matrices (part 1)"),
      "3.2.1b": ("6KIPusOv5fA", "3.2.1 Rotation matrices (part 2)"),
      "3.2.2": ("zJJldJYMxVU", "3.2.2 Angular velocities"),
      "3.2.3a": ("v_KBHaG0mas", "3.2.3 Exponential coordinates of rotation (part 1)"),
      "3.2.3b": ("WHn9xJl43nY", "3.2.3 Exponential coordinates of rotation (part 2)"),
      "3.3.1": ("vlb3P7arbkU", "3.3.1 Homogeneous transformation matrices"),
      "3.3.2a": ("mvGZtO_ruj0", "3.3.2 Twists (part 1)"),
      "3.3.2b": ("VTv0qmLNvjg", "3.3.2 Twists (part 2)"),
      "3.3.3": ("1jYMvm1U2D0", "3.3.3 Exponential coordinates of rigid-body motion"),
      "3.4": ("0wsYPJPGtKE", "3.4 Wrenches")}


def nandi(n):
    return f"[Prof. G. C. Nandi, Lecture {n} — {NANDI_TITLES[n]}]({YT}{NANDI[n]})"


def te(k):
    vid, title = TE[k]
    return f"[ThatsEngineering — {title}]({YT}{vid})"


def mr(k):
    vid, title = MR[k]
    return f"[Modern Robotics {title}]({YT}{vid})"


