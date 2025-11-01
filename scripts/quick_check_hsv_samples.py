#!/usr/bin/env python3
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from holdcheck.color_classifier import classify_color_by_hsv, load_color_ranges

def test(h,s,v,rgb,label):
    colors = load_color_ranges()["colors"]
    c, conf, rule = classify_color_by_hsv(h,s,v,rgb, colors)
    print(f"{label}: HSV({h},{s},{v}) RGB{rgb} -> {c} ({conf:.2f}) | {rule}")

def main():
    # mint인데 unknown으로 찍혔던 샘플들
    for hsv, rgb in [
        ((93,138,162), (74,153,162)),
        ((89,124,118), (61,118,116)),
        ((88,129,109), (54,109,105)),
        ((91,118,116), (62,114,116)),
        ((85,64,103),  (77,103,99)),
        ((88,107,125), (73,125,122)),
        ((90,92,115),  (74,115,115)),
    ]:
        test(*hsv, rgb, "mint?")

    # green인데 mint로 찍혔던
    for hsv, rgb in [
        ((81,81,115),(78,115,104)),
        ((82,70,93),(67,93,86)),
        ((84,102,121),(73,121,111)),
    ]:
        test(*hsv, rgb, "green?")

    # mint로 잘 잡혔던
    test(81,90,103,(67,103,92), "mint OK")

    # blue인데 mint로 찍혔던
    test(98,242,78,(4,58,78), "blue?")

    # orange인데 red로 찍혔던
    test(178,180,255,(255,75,87), "orange?")

if __name__ == '__main__':
    main()


