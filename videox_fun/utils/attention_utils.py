import torch
import os
from typing import Tuple, List

def get_video_mask_token_info(
    hidden_states_shape: torch.Size,
    text_seq_length: int,
    mask_channels: int = 4,
    video_channels: int = 16,
    patch_size: int = 2,
    temporal_compression_ratio: int = 4,
    sample_height: int = 60,
    sample_width: int = 90,
    sample_frames: int = 13
) -> Tuple[List[int], List[int], dict]:
    """
    Calculate token indices for mask and video patches based on actual tensor shapes.
    
    Args:
        hidden_states_shape: Shape of the hidden states tensor [batch, seq_len, dim]
        text_seq_length: Length of text sequence
        mask_channels: Number of mask channels
        video_channels: Number of video channels  
        patch_size: Spatial patch size
        temporal_compression_ratio: Temporal compression ratio
        sample_height: Input video height
        sample_width: Input video width
        sample_frames: Number of video frames
        
    Returns:
        mask_indices: List of mask token indices
        video_indices: List of video token indices
        info_dict: Dictionary with debugging information
    """
    
    # Calculate image token count (excluding text tokens)
    total_seq_length = hidden_states_shape[1]
    image_seq_length = total_seq_length - text_seq_length
    
    # Calculate expected dimensions
    post_patch_height = sample_height // patch_size
    post_patch_width = sample_width // patch_size
    post_time_compression_frames = (sample_frames - 1) // temporal_compression_ratio + 1
    
    expected_patches_per_frame = post_patch_height * post_patch_width
    expected_total_patches = expected_patches_per_frame * post_time_compression_frames
    
    # Total channels (video + mask)
    total_channels = video_channels + mask_channels
    
    # Calculate patches per channel
    patches_per_channel = expected_total_patches // total_channels
    
    # Calculate token counts for each modality
    video_patch_count = patches_per_channel * video_channels
    mask_patch_count = patches_per_channel * mask_channels
    
    # Generate indices (assuming video channels come first, then mask channels)
    video_indices = list(range(0, video_patch_count))
    mask_indices = list(range(video_patch_count, video_patch_count + mask_patch_count))
    
    # Debugging information
    info_dict = {
        'total_seq_length': total_seq_length,
        'text_seq_length': text_seq_length,
        'image_seq_length': image_seq_length,
        'expected_total_patches': expected_total_patches,
        'video_patch_count': video_patch_count,
        'mask_patch_count': mask_patch_count,
        'post_patch_height': post_patch_height,
        'post_patch_width': post_patch_width,
        'post_time_compression_frames': post_time_compression_frames,
        'patches_per_frame': expected_patches_per_frame,
        'total_channels': total_channels,
        'patches_per_channel': patches_per_channel
    }
    
    return mask_indices, video_indices, info_dict


def extract_mask_video_attention(
    attention_probs: torch.Tensor,
    text_seq_length: int,
    mask_indices: List[int],
    video_indices: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract mask-to-video and video-to-mask attention from full attention matrix.
    
    Args:
        attention_probs: Full attention probability matrix [batch, heads, seq_len, seq_len]
        text_seq_length: Length of text sequence
        mask_indices: Indices of mask tokens
        video_indices: Indices of video tokens
        
    Returns:
        mask_to_video_attn: Attention from mask to video tokens [batch, heads, mask_tokens, video_tokens]
        video_to_mask_attn: Attention from video to mask tokens [batch, heads, video_tokens, mask_tokens]
    """
    
    # Extract image-only attention (skip text tokens)
    image_attn_probs = attention_probs[:, :, text_seq_length:, text_seq_length:]
    
    # Extract mask-to-video attention: mask tokens (query) attending to video tokens (key)
    mask_to_video_attn = image_attn_probs[:, :, mask_indices, :][:, :, :, video_indices]
    
    # Extract video-to-mask attention: video tokens (query) attending to mask tokens (key)
    video_to_mask_attn = image_attn_probs[:, :, video_indices, :][:, :, :, mask_indices]
    
    return mask_to_video_attn, video_to_mask_attn


def save_attention_maps(
    mask_to_video_attn: torch.Tensor,
    video_to_mask_attn: torch.Tensor,
    timestep: int,
    block_idx: int,
    save_dir: str,
    token_info: dict = None
) -> None:
    """
    Save attention maps and metadata to disk.
    
    Args:
        mask_to_video_attn: Mask-to-video attention tensor
        video_to_mask_attn: Video-to-mask attention tensor
        timestep: Current timestep
        block_idx: Current block index
        save_dir: Directory to save files
        token_info: Optional token information dictionary
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save attention maps
    mask_to_video_path = os.path.join(save_dir, f'{timestep}_{block_idx}_mask_to_video_attn.pt')
    video_to_mask_path = os.path.join(save_dir, f'{timestep}_{block_idx}_video_to_mask_attn.pt')
    
    torch.save(mask_to_video_attn.cpu(), mask_to_video_path)
    torch.save(video_to_mask_attn.cpu(), video_to_mask_path)
    
    # Save metadata if provided
    if token_info is not None:
        metadata_path = os.path.join(save_dir, f'{timestep}_{block_idx}_metadata.pt')
        torch.save({
            'timestep': timestep,
            'block_idx': block_idx,
            'mask_to_video_shape': mask_to_video_attn.shape,
            'video_to_mask_shape': video_to_mask_attn.shape,
            **token_info
        }, metadata_path)


def visualize_attention_summary(
    mask_to_video_attn: torch.Tensor,
    video_to_mask_attn: torch.Tensor,
    timestep: int,
    block_idx: int
) -> None:
    """
    Print summary statistics of attention maps for debugging.
    
    Args:
        mask_to_video_attn: Mask-to-video attention tensor
        video_to_mask_attn: Video-to-mask attention tensor
        timestep: Current timestep
        block_idx: Current block index
    """
    
    print(f"\n=== Attention Summary - Timestep {timestep}, Block {block_idx} ===")
    print(f"Mask-to-Video Attention Shape: {mask_to_video_attn.shape}")
    print(f"Video-to-Mask Attention Shape: {video_to_mask_attn.shape}")
    print(f"Mask-to-Video - Mean: {mask_to_video_attn.mean().item():.4f}, Std: {mask_to_video_attn.std().item():.4f}")
    print(f"Video-to-Mask - Mean: {video_to_mask_attn.mean().item():.4f}, Std: {video_to_mask_attn.std().item():.4f}")
    print(f"Mask-to-Video - Min: {mask_to_video_attn.min().item():.4f}, Max: {mask_to_video_attn.max().item():.4f}")
    print(f"Video-to-Mask - Min: {video_to_mask_attn.min().item():.4f}, Max: {video_to_mask_attn.max().item():.4f}")
    print("=" * 60)


def setup_mask_video_attention_processor(
    transformer, 
    mask_channels: int = 4,
    video_channels: int = 16,
    save_dir: str = './output_attention',
    patch_size: int = 2,
    temporal_compression_ratio: int = 4,
    sample_height: int = 60,
    sample_width: int = 90,
    sample_frames: int = 13,
    verbose: bool = False
):
    """
    Set up the custom mask-video attention processor for a CogVideoX transformer.
    
    Args:
        transformer: The CogVideoX transformer model
        mask_channels: Number of mask channels (usually 4)
        video_channels: Number of video channels (usually 16) 
        save_dir: Directory to save attention maps
        patch_size: Patch size for the model
        temporal_compression_ratio: Temporal compression ratio
        sample_height: Sample height
        sample_width: Sample width
        sample_frames: Number of frames
        verbose: Whether to print debug information
    
    Returns:
        The transformer with custom attention processor set
    """
    from thirdparty.diffusers.models.attention_processor import CogVideoXMaskVideoAttnProcessor2_0
    
    # Create the custom attention processor
    custom_processor = CogVideoXMaskVideoAttnProcessor2_0(
        mask_channels=mask_channels,
        video_channels=video_channels,
        save_dir=save_dir,
        patch_size=patch_size,
        temporal_compression_ratio=temporal_compression_ratio,
        sample_height=sample_height,
        sample_width=sample_width,
        sample_frames=sample_frames,
        verbose=verbose
    )
    
    # Set the processor for all attention layers
    transformer.set_attn_processor(custom_processor)
    
    if verbose:
        print(f"✅ Set up CogVideoXMaskVideoAttnProcessor2_0")
        print(f"   - Mask channels: {mask_channels}")
        print(f"   - Video channels: {video_channels}")
        print(f"   - Save directory: {save_dir}")
        print(f"   - Will save attention maps during generation")
    
    return transformer


def restore_default_attention_processor(transformer):
    """
    Restore the default CogVideoX attention processor.
    
    Args:
        transformer: The CogVideoX transformer model
    
    Returns:
        The transformer with default attention processor restored
    """
    from thirdparty.diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
    
    # Restore default processor
    default_processor = CogVideoXAttnProcessor2_0()
    transformer.set_attn_processor(default_processor)
    
    print("✅ Restored default CogVideoXAttnProcessor2_0")
    
    return transformer


def compute_mask_video_attention_from_saved_qk(
    query_mask_path: str,
    key_video_path: str,
    query_video_path: str = None,
    key_mask_path: str = None,
    temperature: float = None,
    save_path: str = None
):
    """
    Compute mask-to-video and video-to-mask attention from saved Q, K tensors.
    
    Args:
        query_mask_path: Path to saved mask query tensor [batch, heads, mask_tokens, head_dim]
        key_video_path: Path to saved video key tensor [batch, heads, video_tokens, head_dim] 
        query_video_path: Path to saved video query tensor (for video-to-mask attention)
        key_mask_path: Path to saved mask key tensor (for video-to-mask attention)
        temperature: Temperature for attention computation (if None, uses 1/sqrt(head_dim))
        save_path: Path to save computed attention maps
        
    Returns:
        dict: Dictionary containing attention maps and metadata
    """
    import torch
    import torch.nn.functional as F
    import os
    
    # Load tensors
    query_mask = torch.load(query_mask_path, map_location='cpu')  # [batch, heads, mask_tokens, head_dim]
    key_video = torch.load(key_video_path, map_location='cpu')   # [batch, heads, video_tokens, head_dim]
    
    head_dim = query_mask.shape[-1]
    scale = 1.0 / (head_dim ** 0.5) if temperature is None else temperature
    
    # Compute mask-to-video attention: mask queries attend to video keys
    # query_mask: [batch, heads, mask_tokens, head_dim] 
    # key_video: [batch, heads, video_tokens, head_dim]
    # -> attn_scores: [batch, heads, mask_tokens, video_tokens]
    mask_to_video_scores = torch.matmul(query_mask, key_video.transpose(-2, -1)) * scale
    mask_to_video_attn = F.softmax(mask_to_video_scores, dim=-1)
    
    results = {
        'mask_to_video_attn': mask_to_video_attn,
        'mask_to_video_scores': mask_to_video_scores,
        'metadata': {
            'mask_tokens': query_mask.shape[2],
            'video_tokens': key_video.shape[2],
            'heads': query_mask.shape[1],
            'head_dim': head_dim,
            'scale': scale
        }
    }
    
    # Compute video-to-mask attention if both tensors provided
    if query_video_path and key_mask_path and os.path.exists(query_video_path) and os.path.exists(key_mask_path):
        query_video = torch.load(query_video_path, map_location='cpu')  # [batch, heads, video_tokens, head_dim]
        key_mask = torch.load(key_mask_path, map_location='cpu')       # [batch, heads, mask_tokens, head_dim]
        
        # Video queries attend to mask keys
        # query_video: [batch, heads, video_tokens, head_dim]
        # key_mask: [batch, heads, mask_tokens, head_dim] 
        # -> attn_scores: [batch, heads, video_tokens, mask_tokens]
        video_to_mask_scores = torch.matmul(query_video, key_mask.transpose(-2, -1)) * scale
        video_to_mask_attn = F.softmax(video_to_mask_scores, dim=-1)
        
        results.update({
            'video_to_mask_attn': video_to_mask_attn,
            'video_to_mask_scores': video_to_mask_scores
        })
    
    # Save results if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(results, save_path)
        print(f"Saved computed attention to {save_path}")
    
    return results


def batch_compute_attention_from_saved_tensors(
    attention_dir: str,
    output_dir: str = None,
    timesteps: list = None,
    blocks: list = None,
    verbose: bool = True
):
    """
    Batch compute attention from all saved tensor files in a directory.
    
    Args:
        attention_dir: Directory containing saved Q, K, V tensors
        output_dir: Directory to save computed attention maps
        timesteps: List of timesteps to process (if None, processes all found)
        blocks: List of block indices to process (if None, processes all found)
        verbose: Whether to print progress
        
    Returns:
        dict: Summary of computed attention maps
    """
    import os
    import glob
    
    if output_dir is None:
        output_dir = os.path.join(attention_dir, 'computed_attention')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all query_mask files to get available timesteps and blocks
    query_mask_files = glob.glob(os.path.join(attention_dir, '*_query_mask.pt'))
    
    results_summary = {}
    processed_count = 0
    
    for query_mask_path in query_mask_files:
        # Extract timestep and block from filename
        basename = os.path.basename(query_mask_path)  # e.g., "99_0_query_mask.pt"
        parts = basename.replace('_query_mask.pt', '').split('_')
        
        if len(parts) >= 2:
            timestep, block_idx = parts[0], parts[1]
            
            # Skip if not in requested timesteps/blocks
            if timesteps and timestep not in [str(t) for t in timesteps]:
                continue
            if blocks and block_idx not in [str(b) for b in blocks]:
                continue
            
            # Construct paths for other tensors
            base_path = os.path.join(attention_dir, f'{timestep}_{block_idx}')
            key_video_path = f'{base_path}_key_video.pt'
            query_video_path = f'{base_path}_query_video.pt'
            key_mask_path = f'{base_path}_key_mask.pt'
            
            # Check if required files exist
            if not os.path.exists(key_video_path):
                if verbose:
                    print(f"Missing key_video for timestep {timestep}, block {block_idx}")
                continue
            
            try:
                # Compute attention
                save_path = os.path.join(output_dir, f'{timestep}_{block_idx}_attention.pt')
                
                result = compute_mask_video_attention_from_saved_qk(
                    query_mask_path=query_mask_path,
                    key_video_path=key_video_path,
                    query_video_path=query_video_path if os.path.exists(query_video_path) else None,
                    key_mask_path=key_mask_path if os.path.exists(key_mask_path) else None,
                    save_path=save_path
                )
                
                results_summary[f'{timestep}_{block_idx}'] = {
                    'timestep': timestep,
                    'block_idx': block_idx,
                    'mask_to_video_shape': result['mask_to_video_attn'].shape,
                    'has_video_to_mask': 'video_to_mask_attn' in result,
                    'save_path': save_path
                }
                
                processed_count += 1
                
                if verbose:
                    print(f"✓ Processed timestep {timestep}, block {block_idx}")
                    
            except Exception as e:
                if verbose:
                    print(f"✗ Failed timestep {timestep}, block {block_idx}: {e}")
    
    if verbose:
        print(f"\nProcessed {processed_count} attention computations")
        print(f"Results saved to: {output_dir}")
    
    return results_summary 