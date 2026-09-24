"""Export report.html to PDF with an installed Chromium/Edge browser."""
from __future__ import annotations
import argparse,shutil,subprocess
from pathlib import Path


def browser_path():
    names=("msedge","msedge.exe","chrome","chrome.exe","chromium","chromium.exe")
    for name in names:
        found=shutil.which(name)
        if found:return Path(found)
    candidates=(Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
                Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
                Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"))
    return next((p for p in candidates if p.is_file()),None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html",type=Path);parser.add_argument("pdf",type=Path)
    args=parser.parse_args();browser=browser_path()
    if browser is None:raise SystemExit("Edge/Chrome not found; HTML report remains available")
    args.pdf.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(browser),"--headless","--disable-gpu",
                    f"--print-to-pdf={args.pdf.resolve()}",args.html.resolve().as_uri()],check=True)
    print(args.pdf.resolve())


if __name__=="__main__":main()
