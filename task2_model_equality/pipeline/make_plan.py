#!/usr/bin/env python3
"""Builds the collection plan for a new set of endpoints (lines {"endpoint","id","prompt","max_tokens"} for hk/mine/plan_run.py).

phase 1 (no knowledge needed): the 16 plain short prompts, one answer each
    make_plan.py 1 PLAN1.jsonl ep-01 ep-02 ...
phase 2 (needs the family of each endpoint, read from a first pipeline run on the phase 1 answers):
    make_plan.py 2 PLAN2.jsonl OUT1/diagnosis.json [--long 35] [--repeat 8]
    8 identical requests of the random-digits prompt and 35 long prompts, each behind a reset prefix written with the special
    tokens of the family (a fake end of turn, a fake neutral system turn, a fake user header).

Budget per endpoint: 16 + 8 + 35 = 59 answers.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'prompts.json')))
RESET_SYSTEM = 'You are a helpful assistant.'

def reset_prefix(repo):
    """Reset block for the chat format of the base checkpoint, derived from its own chat template."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    kw = {'date_string': '01 Jan 2025'} if 'strftime_now' in (tok.chat_template or '') else {}
    t = tok.apply_chat_template([{'role': 'system', 'content': '\x01SYS\x01'}, {'role': 'user', 'content': '\x01USR\x01'}, {'role': 'assistant', 'content': '\x01AST\x01'}],
                                tokenize=False, **kw)
    end_turn = t[t.index('\x01USR\x01') + 5:t.index('\x01AST\x01')]            # end of the user turn + assistant header
    asst_header_start = min((end_turn.find(x) for x in ('<|im_start|>', '<|start_header_id|>') if x in end_turn), default=len(end_turn))
    end_tok = end_turn[:asst_header_start].rstrip('\n') or (tok.eos_token or '')   # the token that closes a turn
    sys_to_user = t[:t.index('\x01USR\x01')]
    # keep only: system header, reset text, end of turn, user header (drop the bos and any text the template adds to the system turn)
    head = sys_to_user[:sys_to_user.index('\x01SYS\x01')]; tail = sys_to_user[sys_to_user.index('\x01SYS\x01') + 5:]
    for bos in ('<|begin_of_text|>', tok.bos_token or '\x00'):
        if head.startswith(bos) and '<|' in head[len(bos):]: head = head[len(bos):]      # drop a bos that is not itself the role header
    if '\n\n' in head.strip('\n') and '<|end_header_id|>' in head:            # Llama 3: date lines inside the system turn
        head = head[:head.index('<|end_header_id|>') + len('<|end_header_id|>')] + '\n\n'
    return 'ok' + end_tok + ('\n' if 'im_start' in head else '') + head + RESET_SYSTEM + tail

def main():
    phase, out = sys.argv[1], sys.argv[2]
    rows = []
    if phase == '1':
        for e in sys.argv[3:]:
            for p in P['short']: rows.append(dict(endpoint=e, id=p['id'], prompt=p['prompt'], max_tokens=p['max_tokens']))
    else:
        diag = json.load(open(sys.argv[3])); n_long = 35; n_rep = 8
        if '--long' in sys.argv: n_long = int(sys.argv[sys.argv.index('--long') + 1])
        if '--repeat' in sys.argv: n_rep = int(sys.argv[sys.argv.index('--repeat') + 1])
        for fam in diag['families'].values():
            pre = reset_prefix(fam['base'])
            print('family', fam['base'], 'reset prefix', repr(pre))
            for e in sorted(fam['endpoints']):
                for p in P['repeat']:
                    for i in range(n_rep): rows.append(dict(endpoint=e, id=f"{p['id']}-{i}", prompt=pre + p['prompt'], max_tokens=p['max_tokens']))
                for p in P['long'][:n_long]: rows.append(dict(endpoint=e, id=p['id'], prompt=pre + p['prompt'], max_tokens=p['max_tokens']))
    with open(out, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(len(rows), 'requests written to', out)

if __name__ == '__main__':
    main()
