# ppm_utf16_to_utf8.py
import sys, codecs

def convert(src, dst):
    # 読み込み（BOM 自動判定）
    with codecs.open(src, 'r', 'utf-16') as f:
        text = f.read()

    # PPM 先頭が "P3" または "P6" かチェック
    if not text.lstrip().startswith(('P3', 'P6')):
        raise ValueError("Not a valid PPM file or wrong encoding")

    # 書き込み：UTF‑8 (BOM なし)
    with open(dst, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python ppm_utf16_to_utf8.py in.ppm out.ppm")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
    print("Converted:", sys.argv[1], "→", sys.argv[2])
