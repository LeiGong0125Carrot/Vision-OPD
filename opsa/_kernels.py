"""共享 teacher-forcing 内核: 只取续写段 logits (logits_to_keep), 分块 log_softmax。

相比整段 logits ([L,150k] bf16 可到 2.7GB+), 只保留续写位置 + 128 行分块,
显存峰值降一个量级 —— 卡被其他进程挤占时也能跑。
"""
import torch


def forced_stats(model, inputs, cont_ids, topk=5, chunk=128, want_ent=True, want_topk=True):
    """teacher-force 续写段: 逐 token logp (+entropy, +top-k id/prob), full-vocab。

    返回 (lr[T], ent[T]|None, t5i[T,k]|None, t5p[T,k]|None), 均在 CPU。
    """
    full_ids = torch.cat([inputs["input_ids"], cont_ids.unsqueeze(0).to(model.device)], dim=1)
    kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    if "mm_token_type_ids" in kwargs:
        pad = torch.zeros((1, cont_ids.shape[0]), dtype=kwargs["mm_token_type_ids"].dtype,
                          device=kwargs["mm_token_type_ids"].device)
        kwargs["mm_token_type_ids"] = torch.cat([kwargs["mm_token_type_ids"], pad], dim=1)
    T = cont_ids.shape[0]
    with torch.inference_mode():
        try:
            # 只算最后 T+1 个位置的 lm_head/logits; 行 0..T-1 恰是预测 cont 各 token 的位置
            out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                        logits_to_keep=T + 1, **kwargs)
            logits = out.logits[0, :T]
        except TypeError:
            out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), **kwargs)
            p0 = inputs["input_ids"].shape[1]
            logits = out.logits[0, p0 - 1: p0 - 1 + T]
    lr = torch.empty(T)
    ent = torch.empty(T) if want_ent else None
    t5i = torch.empty((T, topk), dtype=torch.long) if want_topk else None
    t5p = torch.empty((T, topk)) if want_topk else None
    for i in range(0, T, chunk):
        lp = torch.log_softmax(logits[i:i + chunk].float(), dim=-1)
        lr[i:i + chunk] = lp.gather(-1, cont_ids[i:i + chunk, None].to(lp.device)).squeeze(-1).cpu()
        if want_ent:
            ent[i:i + chunk] = (-(lp.exp() * lp).sum(-1)).cpu()
        if want_topk:
            tv, ti = lp.exp().topk(topk, dim=-1)
            t5i[i:i + chunk] = ti.cpu()
            t5p[i:i + chunk] = tv.cpu()
    del logits, out
    return lr, ent, t5i, t5p
