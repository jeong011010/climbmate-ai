#!/usr/bin/env python3
import os, sys, importlib.util
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CC_PATH = os.path.join(ROOT, 'holdcheck', 'color_classifier.py')
spec = importlib.util.spec_from_file_location('color_classifier', CC_PATH)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)  # type: ignore
classify_color_by_hsv = cc.classify_color_by_hsv
load_color_ranges = cc.load_color_ranges

def check(h,s,v,rgb):
    colors = load_color_ranges()["colors"]
    c, conf, reason = classify_color_by_hsv(h,s,v,rgb, colors)
    print(f"HSV({h},{s},{v}) RGB{tuple(rgb)} -> {c} ({conf:.3f}) | {reason}")

def main():
    check(103,129,184,[91,144,184])
    check(101,78,130,[90,115,130])
    check(161,75,220,[220,155,196])

if __name__ == '__main__':
    main()


