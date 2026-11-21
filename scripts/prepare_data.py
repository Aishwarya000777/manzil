"""Decompress the project's *.gz files without deleting the archives."""
import argparse, gzip
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="."); a=p.parse_args()
    files=list(Path(a.root).rglob("*.gz"))
    for src in files:
        dst=src.with_suffix("")
        if dst.exists(): print("skip existing:",dst); continue
        with gzip.open(src,"rb") as inp, open(dst,"wb") as out: out.write(inp.read())
        print("decompressed:",dst)
    print("Processed",len(files),"compressed files.")

if __name__=="__main__": main()
