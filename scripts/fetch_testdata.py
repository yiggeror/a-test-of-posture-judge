#!/usr/bin/env python3
"""Fetch the openly-licensed test images this demo was validated against.

IMPORTANT - read reports/PHASE0.md first. Only TWO of these are full-body
standing shots, and NEITHER is a true lateral (side) view. The sagittal
metrics - which are the main point of the app - are therefore NOT properly
validated. Supply your own side-view photos to change that.

Source: the public `mediapipe-assets` GCS bucket, which is Google's own test
asset bucket for MediaPipe (Apache-2.0 project).
"""
from __future__ import annotations

import os
import sys
import urllib.request

BASE = "https://storage.googleapis.com/mediapipe-assets/"
FILES = {
    # name: (usable?, what it actually is)
    "pose.jpg": (True, "full body, yoga Warrior II lunge - NOT neutral standing"),
    "male_full_height_hands.jpg": (True, "full body frontal standing, cap+mask occlude the ears"),
    "business-person.png": (False, "torso only, ankles out of frame"),
    "portrait.jpg": (False, "head and shoulders only"),
    "test.jpg": (False, "ankles cropped"),
}

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "testdata")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, (usable, desc) in FILES.items():
        dest = os.path.join(OUT, name)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(BASE + name, dest)
        flag = "usable " if usable else "REJECT "
        print(f"{flag} {name:<28} {os.path.getsize(dest):>8} B  {desc}")
    print("\nNo true side-view image is available from any reachable source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
