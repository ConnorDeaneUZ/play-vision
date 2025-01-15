from editor import HighlightDetector

def main():
    detector = HighlightDetector("liverpool-vs-united.mp4")
    detector.process("clips", verbose=False)

if __name__ == "__main__":
    main()
