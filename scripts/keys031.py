import json,sys,urllib.request
g=lambda p: json.load(urllib.request.urlopen("http://127.0.0.1:"+__import__("os").environ.get("STATS","8899")+p))
reqs=g("/kv/keys")["requests"]
for r in reqs: print(r["req"][:50], r["source"], r["num_tokens"], r.get("keys"))
d=g("/kv/keys?req="+reqs[0]["req"])
for x in d["groups"]:
    idx=[r["i"] for r in x["keys"] if r["in_index"]]
    print("group",x["group"],"bt",x["block_tokens"],"keys",x["num_keys"],"hit",x["num_hit_blocks"],"in_index",len(idx),idx)
