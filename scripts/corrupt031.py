import json,sys,urllib.request
g=lambda p: json.load(urllib.request.urlopen("http://127.0.0.1:"+__import__("os").environ.get("STATS","8899")+p))
rid=g("/kv/keys")["requests"][0]["req"]
d=g("/kv/keys?req=%s&group=%s"%(rid,sys.argv[1]))
row=[r for r in d["groups"][0]["keys"] if r["i"]==int(sys.argv[2])][0]
p=row["path"][sys.argv[3] if len(sys.argv)>3 else "0"]
with open(p,"r+b") as f:
    f.seek(4096); b=f.read(1); f.seek(4096); f.write(bytes([b[0]^0xFF]))
print("flipped byte 4096 of", p)
