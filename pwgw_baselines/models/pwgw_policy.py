import abc
import math
import logging

from io import open
from gym import Space
from habitat import Config
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor, device, dtype

from .resnet_encoders import ResNetHintEncoder
from .vision_transformer import vit_base_patch16_224, vit_base_patch32_224
from .utils import CustomFixedCategorical, length2mask

from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import CategoricalNet
from habitat_baselines.common.baseline_registry import baseline_registry

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu,
          "relu": torch.nn.functional.relu,
          "swish": swish
          }


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # [B, heads, L, hidden_size/heads] = [B, 12, L, 64]

    def forward(self, hidden_states, attention_mask, head_mask=None):
        # hidden_states: [B, 60, 768]
        mixed_query_layer = self.query(hidden_states) # [B, 60, 768]
        mixed_key_layer   = self.key(hidden_states) # [B, 60, 768]
        mixed_value_layer = self.value(hidden_states) # [B, 60, 768]

        query_layer = self.transpose_for_scores(mixed_query_layer) # [B, 6, 60, 128]
        key_layer   = self.transpose_for_scores(mixed_key_layer) # [B, 6, 60, 128]
        value_layer = self.transpose_for_scores(mixed_value_layer) # [B, 6, 60, 128]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [B, 6, 60, 60]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # [B, 6, 60, 60]
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask # [B, 6, 60, 60]
        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # [B, 6, 60, 60]

        # NOTE: only for temporary test
        # attention_scores = attention_probs

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs) # [B, 6, 60, 60]

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer) # [B, 6, 60, 128]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [B, 60, 6, 128]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [B, 60, 768]
        context_layer = context_layer.view(*new_context_layer_shape) # [B, 60, 768]

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor) 
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention    = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output       = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions    = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, None if head_mask is None else head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer   = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores


class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.no_lang_ca = config.no_lang_ca # do not update language embeds

        # Language self-attention and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter    = BertIntermediate(config)
        self.lang_output   = BertOutput(config)

        # Vision self-attention and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter    = BertIntermediate(config)
        self.visn_output   = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        if self.no_lang_ca:
            lang_att_output = lang_input
        else:
            lang_att_output, _ = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output, _ = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        if self.no_lang_ca:
            lang_att_output = (lang_input, )
        else:
            lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        if not self.no_lang_ca:
            lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        if self.no_lang_ca:
            lang_output = lang_input
        else:
            lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output[0], visn_att_output[0])

        return lang_output, visn_output


class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_l_layers     = config.num_l_layers
        self.num_r_layers     = config.num_r_layers
        self.num_h_layers     = config.num_h_layers
        self.num_x_layers     = config.num_x_layers
        self.update_lang_bert = config.update_lang_bert

        # Using self.layer instead of self.l_layers to support loading BERT weights.
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.num_l_layers)])
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False

        self.r_layers = nn.ModuleList([BertLayer(config) for _ in range(self.num_r_layers)]) if self.num_r_layers > 0 else None
        self.h_layers = nn.ModuleList([BertLayer(config) for _ in range(self.num_h_layers)]) if self.num_h_layers > 0 else None
        self.x_layers = nn.ModuleList([LXRTXLayer(config) for _ in range(self.num_x_layers)])

    def forward(self, txt_embeds, extended_txt_masks, hist_embeds, extended_hist_masks, img_embeds=None, extended_img_masks=None):
        # text encoding
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()

        # image encoding
        if img_embeds is not None:
            if self.r_layers is not None:
                for layer_module in self.r_layers:
                    temp_output = layer_module(img_embeds, extended_img_masks)
                    img_embeds = temp_output[0]

        # history encoding
        if self.h_layers is not None:
            for layer_module in self.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]
        hist_max_len = hist_embeds.size(1)
        
        # cross-modal encoding
        if img_embeds is None:
            hist_img_embeds = hist_embeds
            extended_hist_img_masks = extended_hist_masks
        else:
            hist_img_embeds = torch.cat([hist_embeds, img_embeds], 1)
            extended_hist_img_masks = torch.cat([extended_hist_masks, extended_img_masks], -1)
        
        for layer_module in self.x_layers:
            txt_embeds, hist_img_embeds = layer_module(txt_embeds, extended_txt_masks, hist_img_embeds, extended_hist_img_masks)

        hist_embeds = hist_img_embeds[:, :hist_max_len]
        if img_embeds is not None:
            img_embeds = hist_img_embeds[:, hist_max_len:]
            
        return txt_embeds, hist_embeds, img_embeds


class RGBEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        stem_channel = config.RGB_ENCODER.stem_channel
        
        # for input size: [B, 3, 224, 224]
        self.stem_conv1 = nn.Conv2d(3, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        
        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        
        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

    def forward(self, observations):
        rgb_image = observations["rgb"] # [B, 224, 224, 3]
        rgb_image = rgb_image.permute(0, 3, 1, 2) # [B, 3, 224, 224]
        rgb_image = rgb_image.contiguous() / 255.0 # normalize rgb image
        return rgb_image

        rgb_embed = self.stem_conv1(rgb_image)
        rgb_embed = self.stem_relu1(rgb_embed)
        rgb_embed = self.stem_norm1(rgb_embed)
        
        rgb_embed = self.stem_conv2(rgb_embed)
        rgb_embed = self.stem_relu2(rgb_embed)
        rgb_embed = self.stem_norm2(rgb_embed)
        
        rgb_embed = self.stem_conv3(rgb_embed)
        rgb_embed = self.stem_relu3(rgb_embed)
        rgb_embed = self.stem_norm3(rgb_embed)

        return rgb_embed # [B, stem_channel, 224, 224]


class DepthEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        stem_channel = config.DEPTH_ENCODER.stem_channel
        
        # for input size: [B, 3, 224, 224]
        self.stem_conv1 = nn.Conv2d(1, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.stem_relu1 = nn.GELU()
        self.stem_norm1 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        
        self.stem_conv2 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu2 = nn.GELU()
        self.stem_norm2 = nn.BatchNorm2d(stem_channel, eps=1e-5)
        
        self.stem_conv3 = nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.stem_relu3 = nn.GELU()
        self.stem_norm3 = nn.BatchNorm2d(stem_channel, eps=1e-5)

    def forward(self, observations):
        depth_image = observations["depth"] # [B, 224, 224, 1]
        depth_image = depth_image.permute(0, 3, 1, 2) # [B, 1, 224, 224]
        return depth_image

        depth_embed = self.stem_conv1(depth_image)
        depth_embed = self.stem_relu1(depth_embed)
        depth_embed = self.stem_norm1(depth_embed)
        
        depth_embed = self.stem_conv2(depth_embed)
        depth_embed = self.stem_relu2(depth_embed)
        depth_embed = self.stem_norm2(depth_embed)
        
        depth_embed = self.stem_conv3(depth_embed)
        depth_embed = self.stem_relu3(depth_embed)
        depth_embed = self.stem_norm3(depth_embed)

        return depth_embed # [B, stem_channel, 224, 224]
    

class HintEncoder(nn.Module):
    def __init__(self, config, observation_space):
        super().__init__()
        self.hint_image_encoder = ResNetHintEncoder(observation_space=observation_space,
                                                    output_size=config.HINT_ENCODER.output_size, 
                                                    resnet_output_size=config.HINT_ENCODER.resnet_output_size, 
                                                    device=device,
                                                    spatial_output=True)
        self.hint_image_conv = nn.Conv1d(self.hint_image_encoder.output_shape[0], config.HINT_ENCODER.output_size, 1)
        self.hint_image_linear = nn.Linear(config.HINT_ENCODER.output_size, config.hidden_size)
        self.hint_image_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.hint_position_linear = nn.Linear(5, config.hidden_size)
        self.hint_position_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, observations):
        # hint image encoding, hint image: [B, 224, 224, C]
        hint_image_embed = self.hint_image_encoder(observations) # [B, 2048+64, 4, 4]
        hint_image_embed = torch.flatten(hint_image_embed, 2) # [B, 2048+64, 16]
        hint_image_embed = self.hint_image_conv(hint_image_embed) # [B, 256, 16]
        hint_image_embed = hint_image_embed.permute(0, 2, 1) # [B, 16, 256]
        hint_image_embed = self.hint_image_norm(self.hint_image_linear(hint_image_embed)) # [B, 16, 768]
        hint_image_embed = torch.mean(hint_image_embed, 1) # [B, 768]
        # hint_image_embed = torch.zeros_like(hint_image_embed) # NOTE: for ablation study

        # hint position encoding
        _, H, W, _ = observations["rgb"].size() # H=672, W=896
        B, h, w, _ = observations["hint"].size() # h=224, w=224
        hint_position = observations["hint_position"] # [B, 4]
        device = hint_position.device

        divisor = torch.tensor([W, H, W, H], dtype=hint_position.dtype, device=device) # [4]
        hint_position = hint_position / divisor # [B, 4]

        normalized_area = torch.tensor([[(w * h) / float(W * H)]], dtype=hint_position.dtype, device=device) # [1, 1]
        normalized_area = normalized_area.expand(B, 1) # [B, 1]
        
        hint_position = torch.cat([hint_position, normalized_area], dim=1) # [B, 5]
        hint_position_embed = self.hint_position_norm(self.hint_position_linear(hint_position)) # [B, 768]
        # hint_position_embed = torch.zeros_like(hint_position_embed) # NOTE: for ablation study

        hint_embed = hint_image_embed + hint_position_embed # [B, 768]
        hint_embed = self.layer_norm(hint_embed)
        hint_embed = self.dropout(hint_embed)

        return hint_embed
    

class HintLanguageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        vocab_size = config.HINT_ENCODER.language_vocab_size

        self.embedding = nn.Embedding(vocab_size, config.hidden_size)
        self.language_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.hint_position_linear = nn.Linear(5, config.hidden_size)
        self.hint_position_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, observations):
        hint_language = observations["hint_language"].long().squeeze(-1)
        language_embed = self.language_norm(self.embedding(hint_language))

        _, H, W, _ = observations["rgb"].size()
        B = hint_language.size(0)
        hint_position = observations["hint_position"]
        device = hint_position.device

        divisor = torch.tensor([W, H, W, H], dtype=hint_position.dtype, device=device)
        hint_position = hint_position / divisor

        normalized_area = torch.ones((B, 1), dtype=hint_position.dtype, device=device)
        hint_position = torch.cat([hint_position, normalized_area], dim=1)
        hint_position_embed = self.hint_position_norm(self.hint_position_linear(hint_position))
        # hint_position_embed = torch.zeros_like(hint_position_embed) # NOTE: for ablation study

        hint_embed = language_embed + hint_position_embed
        hint_embed = self.layer_norm(hint_embed)
        hint_embed = self.dropout(hint_embed)

        return hint_embed
    

class HistoryEmbedding(nn.Module):
    def __init__(self, config, num_actions):
        super().__init__()
        self.num_actions       = num_actions

        self.visual_linear     = nn.Linear(config.vit_embed_dim, config.hidden_size)
        self.visual_layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.action_linear     = nn.Linear(num_actions, config.hidden_size)
        self.action_layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps) # [B, 768]

        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout    = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visual_embed, selected_action):
        visual_embed = self.visual_layer_norm(self.visual_linear(visual_embed)) # [B, 768]
        visual_embed = torch.mean(visual_embed, 1) # [B, 768]

        action_embed = F.one_hot(selected_action.squeeze(-1), num_classes=self.num_actions).to(torch.float32) # [B, num_actions]
        action_embed = self.action_layer_norm(self.action_linear(action_embed)) # [B, 768]

        embeddings = visual_embed # [B, 768]
        embeddings = self.layer_norm(embeddings) # [B, 768]
        embeddings = self.dropout(embeddings) # [B, 768]

        return embeddings
    

class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        # compute the positional encoding once
        pos_enc = torch.zeros(config.max_hist_len, config.hidden_size) # [10, 768]
        pos = torch.arange(0, config.max_hist_len, dtype=torch.float).unsqueeze(1) # [10, 1]
        div_term = torch.exp(torch.arange(0, config.hidden_size, 2).float() * (-math.log(10000.0) / config.hidden_size)) # [384]
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0) # [1, 10, 768]

        # register the positional encoding as a buffer to avoid it being considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x


class NextActionPrediction(nn.Module):
    def __init__(self, config, num_actions):
        super().__init__()
        self.action_embedding = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.ReLU(),
                                              BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                                              nn.Dropout(config.pred_head_dropout_prob))
        
        self.action_distribution = CategoricalNet(config.hidden_size, num_actions) # this is for IL policy

    def forward(self, feature):
        # [B, 10+49+1, 768]
        action_embedding = self.action_embedding(feature)
        action_distribution = self.action_distribution(action_embedding)
        return action_distribution


class NavCMT(nn.Module):
    def __init__(self, config: Config, observation_space: Space, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.patch_grid_size = int(config.vit_input_size / config.vit_patch_size) # 224 / 32 = 7

        self.rgb_encoder   = RGBEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        if getattr(config.HINT_ENCODER, "input_type", "image") == "language":
            self.hint_encoder = HintLanguageEncoder(config)
        else:
            self.hint_encoder = HintEncoder(config, observation_space)
        self.hist_embed    = HistoryEmbedding(config, num_actions)
        self.pos_embed     = PositionalEmbedding(config)

        self.spatial_transformer = vit_base_patch32_224(pretrained=False,
                                                        in_chans=config.RGB_ENCODER.stem_channel+config.DEPTH_ENCODER.stem_channel,
                                                        drop_rate=config.hidden_dropout_prob,
                                                        attn_drop_rate=config.attention_probs_dropout_prob,
                                                        drop_path_rate=0.0)

        self.temporal_transformer = nn.ModuleList([BertLayer(config) for _ in range(config.num_t_layers)]) if config.num_t_layers > 0 else None

        # for next action prediction
        self.next_action = NextActionPrediction(config, num_actions)

        # only used for BertPreTrainedModel
        # self.init_weights()

    def forward(self, mode: str, observations=None, selected_action=None, hist_embeds=None, hist_lens=None):
        if mode == "history":
            if selected_action is None: 
                rgb_embed    = self.rgb_encoder(observations) # [B, 3, 224, 224]
                # rgb_embed    = torch.zeros_like(rgb_embed) # NOTE: for ablation study
                depth_embed  = self.depth_encoder(observations) # [B, 1, 224, 224]
                # depth_embed  = torch.zeros_like(depth_embed) # NOTE: for ablation study
                visual_embed = torch.cat([rgb_embed, depth_embed], 1) # [B, 3+1, 224, 224]
                hint_embed   = self.hint_encoder(observations) # [B, 768]

                hint_embed, visual_embed = self.spatial_transformer.forward_features_plus(hint_embed, visual_embed) # hint_embed: [B, 768], visual_embed: [B, 49, 768]
                observations = visual_embed # [B, 49, 768]
                selected_action = torch.zeros(observations.size(0), dtype=torch.int64, device=observations.device) # [B]

            hist_embed = self.hist_embed(observations, selected_action) # [B, 768]
            return hist_embed
        elif mode == "observation":
            rgb_embed    = self.rgb_encoder(observations) # [B, 3, 224, 224]
            # rgb_embed    = torch.zeros_like(rgb_embed) # NOTE: for ablation study
            depth_embed  = self.depth_encoder(observations) # [B, 1, 224, 224]
            # depth_embed  = torch.zeros_like(depth_embed) # NOTE: for ablation study
            visual_embed = torch.cat([rgb_embed, depth_embed], 1) # [B, 3+1, 224, 224]
            hint_embed   = self.hint_encoder(observations) # [B, 768]

            hint_embed, visual_embed = self.spatial_transformer.forward_features_plus(hint_embed, visual_embed) # hint_embed: [B, 768], visual_embed: [B, 49, 768]
            hint_embed = hint_embed.unsqueeze(1) # hint_embed: [B, 1, 768]
            visual_embed_spatial = visual_embed

            hist_embeds = torch.stack(hist_embeds, 1) # [B, hist_len, 768]
            hist_embeds = self.pos_embed(hist_embeds)
            hist_visual_hint_embed = torch.cat([hist_embeds, visual_embed, hint_embed], 1) # [B, 10+49+1, 768]

            hist_masks = length2mask(hist_lens, size=hist_embeds.size(1), dtype=hist_embeds.dtype) # [B, 1, 1, hist_len]

            visual_len = [visual_embed.size(1) for _ in range(visual_embed.size(0))]
            visual_mask = length2mask(visual_len, dtype=visual_embed.dtype) # [B, 1, 1, 49]

            hint_len = [hint_embed.size(1) for _ in range(hint_embed.size(0))]
            hint_mask = length2mask(hint_len, dtype=hint_embed.dtype) # [B, 1, 1, 1]

            hist_visual_hint_masks = torch.cat([hist_masks, visual_mask, hint_mask], -1) # [B, 1, 1, hist_len+49+1]
            hist_visual_hint_masks = hist_visual_hint_masks.to(hist_visual_hint_embed.device)

            # NOTE: for ablation study
            for layer in self.temporal_transformer:
                layer_output = layer(hist_visual_hint_embed, hist_visual_hint_masks)
                hist_visual_hint_embed = layer_output[0] # [B, L, 768]
                attention_output = layer_output[1] # [B, num_attention_heads, L, L]
            # attention_output = None # NOTE: for ablation study

            hint_embed = hist_visual_hint_embed[:, -1] # [B, 768]

            # using the next action prediction module to predict the final action
            action_distribution = self.next_action(hint_embed)

            return visual_embed_spatial, hist_visual_hint_embed, attention_output, action_distribution


class ILPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        super(Policy, self).__init__()
        # this can be used for mode == "history"
        self.net = net

    # inference: only for mode == "observation"
    def act(self, mode, observations=None, selected_action=None, hist_embeds=None, hist_lens=None, deterministic=False):
        visual_embed_spatial, hist_visual_hint_embed, attention_output, action_distribution = self.net(mode, observations, selected_action, hist_embeds, hist_lens)

        hist_max_len = len(hist_embeds)
        agent_state = hist_visual_hint_embed[:, 0] # [CLS] token
        hist_embeds = hist_visual_hint_embed[:, :hist_max_len]
        visual_embed_temporal = hist_visual_hint_embed[:, hist_max_len:-1]
        hint_embed = hist_visual_hint_embed[:, -1] # [B, 768]

        if deterministic:
            action = action_distribution.mode()
        else:
            action = action_distribution.sample()

        return visual_embed_spatial, action
        # return visual_embed_spatial, action, attention_output # NOTE: only for temporary test

    # training: only for mode == "observation"
    def build_distribution(self, mode, observations=None, selected_action=None, hist_embeds=None, hist_lens=None) -> CustomFixedCategorical:
        visual_embed_spatial, hist_visual_hint_embed, attention_output, action_distribution = self.net(mode, observations, selected_action, hist_embeds, hist_lens)
        
        # NOTE: for ablation study
        hist_max_len = len(hist_embeds)
        hint2visual_attention = attention_output[:, :, -1, hist_max_len:-1] # [B, 12, 49]
        visual_attention_map = torch.mean(hint2visual_attention, 1) # [B, 49]
        visual_attention_map = visual_attention_map.reshape(-1, self.net.patch_grid_size, self.net.patch_grid_size) # [B, 7, 7]
        # visual_attention_map = None # NOTE: for ablation study

        return visual_embed_spatial, visual_attention_map, action_distribution
    

@baseline_registry.register_policy
class PWGWPolicy(ILPolicy):
    def __init__(self, model_config: Config, observation_space: Space, action_space: Space) -> None:
        model = NavCMT(config=model_config, observation_space=observation_space, num_actions=action_space.n)
        super().__init__(model, action_space.n)

    @classmethod
    def from_config(cls, config: Config, observation_space: Space, action_space: Space):
        return cls(model_config=config.MODEL, observation_space=observation_space, action_space=action_space)
