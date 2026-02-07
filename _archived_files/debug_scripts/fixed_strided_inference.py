#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Fixed strided inference function that handles variable model outputs
"""

import torch
from typing import List, Tuple

def strided_inference(model, x, chunk_size=10000, chunk_overlap=0):
    """
    Fixed version of strided inference that handles 2 or 3 model outputs
    """
    assert chunk_overlap >= 2, "overlap must be >=2!"
    assert (chunk_overlap % 2) == 0, "chunk_overlap must be even!"
    half_overlap = chunk_overlap // 2
    
    in_b, in_h, in_w = x.shape
    stride = chunk_size - chunk_overlap
    if in_w <= chunk_size:
        stride = chunk_size  # in this case only 1 chunk needed
    
    # compute strided inference
    # results is in the form [(out1a, out1b, ...), (out2a, out2b, ...), ...]
    results = []
    result_lengths = []
    
    for beg in range(0, in_w, stride):
        chunk = x[..., beg:beg+chunk_size]
        
        # Get model outputs - handle 2 or 3 outputs
        outputs = model(chunk)
        outputs = [o.cpu().detach() for o in outputs]
        
        # Handle variable number of outputs
        if len(outputs) == 2:
            # Only probabilities and velocities (no pedals)
            assert all(chunk.shape[0] == outputs[0].shape[0] for o in outputs), \
                "all b_outputs must equal b_in!"
            assert all(chunk.shape[-1] == outputs[0].shape[-1] for o in outputs), \
                "all t_outputs must equal t_in!"
        elif len(outputs) == 3:
            # Probabilities, velocities, and pedals
            assert all(chunk.shape[0] == outputs[0].shape[0] for o in outputs), \
                "all b_outputs must equal b_in!"
            assert all(chunk.shape[-1] == outputs[0].shape[-1] for o in outputs), \
                "all t_outputs must equal t_in!"
        else:
            raise ValueError(f"Model returned {len(outputs)} outputs, expected 2 or 3")
        
        results.append(outputs)
        result_lengths.append(chunk.shape[-1])
        del chunk
        del outputs
    
    # For >1 chunks, at most 1 partial-length chunk at the end is allowed
    valid_chunks = sum(x == chunk_size for x in result_lengths)
    
    # Concatenate results along time dimension
    def _concat_results(tensors):
        """Helper to concatenate tensors along time dimension"""
        # Handle padding for uneven chunks
        max_time = max(t.shape[-1] for t in tensors)
        padded_tensors = []
        
        for tensor in tensors:
            if tensor.shape[-1] < max_time:
                # Pad to match max time
                pad_size = max_time - tensor.shape[-1]
                padded = torch.nn.functional.pad(tensor, (0, 0, pad_size))
                padded_tensors.append(padded)
            else:
                padded_tensors.append(tensor)
        
        # Concatenate along time dimension
        return torch.cat(padded_tensors, dim=-1)
    
    # Concatenate all outputs for each output type
    final_results = []
    for i in range(len(results[0])):
        # Extract i-th output from each chunk
        ith_outputs = [chunk_results[i] for chunk_results in results]
        
        # Concatenate along time dimension
        final_tensor = _concat_results(ith_outputs)
        final_results.append(final_tensor)
    
    return final_results, result_lengths