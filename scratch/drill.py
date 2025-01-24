import json
import os
from copy import deepcopy

import tqdm

from lat_alignment import ixb

host = "169.254.1.1"
port = 4545
sock, send, recv = ixb.init(host, port)

template: dict
if os.path.isfile("Tightening_Tighten_Template.json"):
    with open("Tightening_Tighten_Template.json") as f:
        template = json.load(f)
else:
    send(ixb.construct_2501(1))
    mid, rev, dat, d, t = recv()
    info, template = ixb.decode2501(mid, dat)
    with open("Tightening_Tighten_Template.json", "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=4)

adjs = []
for r in range(1, 10):
    for c in range(1, 10):
        for a in range(1, 6):
            adjs += [f"P{r}{c}V{a}"]

failed = []
for i, adj in enumerate(tqdm.tqdm(adjs)):
    prog = deepcopy(template)
    prog["name"] = adj
    prog["indexId"]["value"] = i + 1
    prog["steps"][1]["stepTightenToAngle"]["angleTarget"] = 30
    send(ixb.construct_2500(i + 1, prog))
    mid, rev, dat, d, t = recv()
    if mid == "0004" or d or t:
        failed += [adjs]
        sock, send, recv = ixb.init(host, port)
print(f"failed: {failed}")

ixb.close(sock, send)
