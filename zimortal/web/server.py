"""Run with python -m zimortal.web.server; no extra dependencies required."""

import argparse
import json
import random
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from zimortal.engine import RuleClarificationRequired, RuleEngine, meld_huxi
from zimortal.engine.types import ActionType as A

from .layout import arrange_hand

LABELS = {
    "draw": "摸牌",
    "discard": "出牌",
    "chi": "吃",
    "peng": "碰",
    "wei": "偎",
    "stinky_wei": "臭偎",
    "pao": "跑",
    "ti": "提",
    "hu": "胡",
    "pass": "过",
}


def discard_records(history):
    """Historical discards, including cards subsequently claimed by a player."""
    records = [[] for _ in range(3)]
    pending = None
    for step, action in enumerate(history, 1):
        if action.kind == A.DISCARD:
            pending = {
                "tile": action.tile,
                "step": step,
                "status": "pending",
                "claimed_by": None,
                "claim_kind": None,
            }
            records[action.player].append(pending)
        elif pending is not None and action.source_type and action.source_type.value == "discard":
            if action.kind in (A.CHI, A.PENG, A.PAO):
                pending.update(
                    status="claimed", claimed_by=action.player, claim_kind=action.kind.value
                )
                pending = None
            elif action.kind == A.PASS and action.forced:
                pending["status"] = "landed"
                pending = None
    return records


def snapshot(engine, state):
    error = None
    try:
        legal = engine.legal_actions(state)
    except RuleClarificationRequired as exc:
        legal = ()
        error = str(exc)
    discards = discard_records(state.history)
    return {
        "players": [
            {
                "discards": discards[seat],
                "last_action": None,
                "hand": sorted(p.hand),
                "columns": arrange_hand(p.hand, p.kans),
                "kans": sorted(p.kans),
                "melds": [asdict(m) | {"huxi": meld_huxi(m)} for m in p.melds],
                "huxi": sum(meld_huxi(m) for m in p.melds)
                + sum(6 if t >= 10 else 3 for t in p.kans),
                "passed_peng": sorted(p.passed_peng),
                "passed_chi": sorted(p.passed_chi),
                "hu_disabled": p.hu_disabled,
                "quad_count": p.quad_count,
            }
            for seat, p in enumerate(state.players)
        ],
        "remaining": len(state.deck),
        "river": list(state.river),
        "pending": asdict(state.pending) if state.pending else None,
        "phase": state.phase,
        "turn": state.turn,
        "actor": legal[0].player if legal else None,
        "legal": [asdict(a) for a in legal],
        "terminal": state.terminal,
        "winner": state.winner,
        "settlement": asdict(state.settlement) if state.settlement else None,
        "error": error,
    }


def explain(action, before, after):
    if action.kind == A.DRAW:
        return (
            "墩牌公开后，按偎／提、胡、跑、碰、吃的优先级处理。"
            if after.pending
            else "牌堆已空，本局结束。"
        )
    if action.kind == A.HU:
        return "胡牌结构成立且达到15胡；天胡为起手例外，其余胡牌来自墩牌。"
    if action.kind == A.CHI:
        return "墩牌先由摸牌者决定吃或过，再轮到下家；弃牌只能由下家吃。该方案已完成全部下比，且能完成本次出牌或满足夹比。"
    if action.kind == A.PASS:
        if before.phase == "opening":
            return "开局检查结束，由庄家出第一张。"
        if before.phase == "post_meld":
            return "强制进张后的胡牌检查结束，按出牌义务继续。"
        if action.forced:
            return "无人接牌，来牌落桌，由来牌者的下家摸牌。"
        return "主动放弃当前优先级的动作；放弃碰记录过碰，放弃吃记录过张，放弃胡不连带放弃碰或吃。"
    if action.kind == A.DISCARD:
        return "从未锁定的手牌中出一张，其他玩家按优先级响应；弃牌不能胡，自己打过的牌以后不能吃。"
    if action.source_type and action.source_type.value == "initial":
        return "起手四张相同，强制提；起手双提的下一次进张免出牌。"
    if action.kind in (A.WEI, A.STINKY_WEI):
        return "自摸第三张必须偎，过碰后为臭偎；强制动作先于胡牌，再检查能否胡。"
    if action.kind == A.TI:
        return "自摸第四张必须提，优先于胡牌；新牌组胡息替换原胡息。"
    if action.kind == A.PAO:
        return "第四张形成跑；墩牌先检查各家的胡牌机会，再执行跑。"
    return "碰可选择放弃，优先于吃；碰后按出牌义务继续。"


def build_game(seed=118, dealer=0):
    engine = RuleEngine()
    initial = engine.new_game(seed, dealer)
    state = initial
    rng = random.Random(seed)
    last_actions = [None, None, None]
    frames = [
        snapshot(engine, initial)
        | {
            "index": 0,
            "action": None,
            "note": "发牌完成，先处理起手提与庄家天胡。",
            "verified": True,
        }
    ]
    for index in range(1, 1001):
        if state.terminal or frames[-1]["error"]:
            break
        legal = engine.legal_actions(state)
        action = rng.choice(legal)
        before = state
        state = engine.step(state, action)
        state.validate()
        frame = snapshot(engine, state)
        shown = asdict(action)
        if action.kind == A.DRAW and state.pending:
            shown["tile"] = state.pending.tile
        last_actions[action.player] = shown
        for seat, player in enumerate(frame["players"]):
            player["last_action"] = last_actions[seat]
        frame.update(
            index=index, action=shown, note=explain(action, before, state), verified=action in legal
        )
        frames.append(frame)
    else:
        raise RuntimeError("simulation exceeded 1000 actions")
    replay = engine.replay(initial, state.history)
    if replay.serialize() != state.serialize():
        raise RuntimeError("replay mismatch")
    return {
        "seed": seed,
        "dealer": dealer,
        "frames": frames,
        "replay_verified": True,
        "actions": len(state.history),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/game":
            try:
                args = parse_qs(url.query)
                seed = int(args.get("seed", ["118"])[0])
                dealer = int(args.get("dealer", ["0"])[0])
                if not -(2**31) <= seed < 2**31 or dealer not in range(3):
                    raise ValueError("seed/dealer out of range")
                body = json.dumps(build_game(seed, dealer), ensure_ascii=False).encode()
            except ValueError as exc:
                self.respond(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            except (RuntimeError, AssertionError, TypeError):
                self.respond(
                    500, b'{"error":"simulation failed; see server log"}', "application/json"
                )
                import traceback

                traceback.print_exc()
                return
            self.respond(200, body, "application/json; charset=utf-8")
        elif url.path in ("/", "/index.html"):
            self.respond(
                200, Path(__file__).with_name("index.html").read_bytes(), "text/html; charset=utf-8"
            )
        else:
            self.respond(404, b"Not found", "text/plain")

    def respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="ZiMortal local game review")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ZiMortal review: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
