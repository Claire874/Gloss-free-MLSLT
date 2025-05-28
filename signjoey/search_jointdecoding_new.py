import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from signjoey.decoders import Decoder, TransformerDecoder
from signjoey.embeddings import Embeddings
from signjoey.helpers import tile



def joint_beam_search(
    decoder: Decoder,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    embed: Embeddings,
    ctc_layer,
    n_best: int = 1,
) -> (np.array, np.array):
    """
    Joint CTC/Attention Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param decoder:
    :param size: size of the beam
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param embed:
    :param ctc_layer: CTC layer for computing CTC scores
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    assert size > 0, "Beam size must be >0."
    assert n_best <= size, "Can only return {} best hypotheses.".format(size)

    transformer = isinstance(decoder, TransformerDecoder)
    batch_size = src_mask.size(0)
    att_vectors = None  # not used for Transformer

    if not transformer:
        hidden = decoder._init_hidden(encoder_hidden)
    else:
        hidden = None

    if hidden is not None:
        hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size

    encoder_output = tile(encoder_output.contiguous(), size, dim=0)  # batch*k x src_len x enc_hidden_size
    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

    if transformer:
        trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
    else:
        trg_mask = None

    batch_offset = torch.arange(batch_size, dtype=torch.long, device=encoder_output.device)
    beam_offset = torch.arange(0, batch_size * size, step=size, dtype=torch.long, device=encoder_output.device)
    alive_seq = torch.full([batch_size * size, 1], bos_index, dtype=torch.long, device=encoder_output.device)
    topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
    topk_log_probs[:, 1:] = float("-inf")
    hypotheses = [[] for _ in range(batch_size)]
    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):
        #print(f"Step {step} - Batch size: {batch_size}, Beam size: {size}")

        if transformer:
            decoder_input = alive_seq
        else:
            decoder_input = alive_seq[:, -1].view(-1, 1)

        trg_embed = embed(decoder_input)
        logits, hidden, att_scores, att_vectors = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=trg_embed,
            hidden=hidden,
            prev_att_vector=att_vectors,
            unroll_steps=1,
            trg_mask=trg_mask,
        )

        if transformer:
            logits = logits[:, -1]
            hidden = None

        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        curr_scores = curr_scores.view(batch_size, size, -1)
        #print(f"curr_scores shape: {curr_scores.shape}")
        topk_scores, topk_ids = curr_scores.view(batch_size, -1).topk(size, dim=-1)
        #print(f"topk_scores : {topk_scores}")
        #print(f"topk_ids shape: {topk_ids.shape}")

        if alpha > -1:
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

        topk_beam_index = topk_ids.div(decoder.output_size)
        topk_ids = topk_ids.fmod(decoder.output_size)
        batch_index = topk_beam_index + beam_offset[: topk_beam_index.size(0)].unsqueeze(1)
        select_indices = batch_index.view(-1).long()
        alive_seq = torch.cat([alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1)

        # Joint scoring with CTC
        ctc_log_probs = ctc_layer(encoder_output).log_softmax(dim=-1)
        #print(f"ctc_log_probs shape before reshaping: {ctc_log_probs.shape}")

        # Calculate the correct sequence length
        seq_len = ctc_log_probs.size(1)
        vocab_size = ctc_log_probs.size(2)

        # Ensure the size is valid before reshaping
        expected_size = batch_size * size * seq_len * vocab_size
        assert ctc_log_probs.numel() == expected_size, (
            f"Expected size {expected_size}, but got {ctc_log_probs.numel()}."
        )

        # Reshape ctc_log_probs correctly
        ctc_log_probs = ctc_log_probs.view(batch_size * size, seq_len, vocab_size)
        #print(f"ctc_log_probs reshaped: {ctc_log_probs.shape}")

        # Gather CTC scores
        topk_ctc_scores = torch.gather(ctc_log_probs, 2, topk_ids.view(-1, 1, 1).expand(-1, seq_len, 1)).squeeze(-1).sum(dim=1)
        ctc_mean = topk_ctc_scores.mean()
        ctc_std = topk_ctc_scores.std() + 1e-8#1e-9  # Add epsilon to avoid division by zero
        norm_ctc_scores = (topk_ctc_scores - ctc_mean) / ctc_std

        # Z-score normalization of attention scores
        att_mean = topk_scores.mean()
        att_std = topk_scores.std() + 1e-8  # Add epsilon to avoid division by zero
        norm_att_scores = (topk_scores - att_mean) / att_std
        #print(norm_att_scores,norm_ctc_scores)
      

        # Penalize <pad> tokens
        pad_penalty = (alive_seq[:, 1:] == pad_index).float().sum(dim=1).view(batch_size, size) * -1e9

        combined_scores = norm_att_scores + 0.3 * norm_ctc_scores.view(batch_size, size) #+ pad_penalty
        #norm_att_scores + 0.3 * norm_ctc_scores.view(batch_size, size) + pad_penalty


        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        end_condition = is_finished[:, 0].eq(True)

        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(True)
                finished_hyp = is_finished[i].nonzero(as_tuple=True)[0]
                for j in finished_hyp:
                    # Ignore hypotheses that contain <pad> tokens
                    if pad_index not in predictions[i, j, 1:]:
                        if (predictions[i, j, 1:] == eos_index).nonzero(as_tuple=True)[0].numel() < 2:
                            hypotheses[b].append(
                                (combined_scores[i, j], predictions[i, j, 1:])
                            )
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(False).nonzero(as_tuple=True)[0]
            if len(non_finished) == 0:
                break

            # Adjust batch size and beam size for non-finished batches
            batch_size = non_finished.size(0)

            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))

            # Update beam_offset for the new batch size
            beam_offset = torch.arange(0, batch_size * size, step=size, dtype=torch.long, device=encoder_output.device)

        select_indices = batch_index.view(-1).long()
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

        if hidden is not None and not transformer:
            if isinstance(hidden, tuple):
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                hidden = hidden.index_select(1, select_indices)

        if att_vectors is not None:
            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = (
            np.ones((len(hyps), max([h.shape[0]+1 for h in hyps])), dtype=int) * pad_value
        )
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
                filled[j, k+1] = eos_index
        #print('hyps', str(hyps))
        #print('filled', str(filled))
        return filled

    assert n_best == 1
    final_outputs = pad_and_stack_hyps(
        [r[0].cpu().numpy() for r in results["predictions"]], pad_value=pad_index
    )

    return final_outputs, None

